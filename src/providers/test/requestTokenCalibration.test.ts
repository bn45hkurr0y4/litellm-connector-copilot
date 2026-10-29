import * as assert from "assert";
import * as sinon from "sinon";
import * as vscode from "vscode";
import * as tokenUtils from "../../adapters/tokenUtils";
import {
    getCalibrationFactor,
    recordCalibrationObservation,
    resetCalibration,
} from "../../adapters/tokenizers/tokenCalibration";
import { StreamTokenCapture } from "../../adapters/streaming/streamTokenCapture";
import { ConfigManager } from "../../config/configManager";
import { createMockModel, createMockSecrets, noop } from "../../test/utils/testMocks";
import { RequestBuilder } from "../base/requestBuilder";
import { LiteLLMChatProvider } from "../liteLLMChatProvider";
import { LiteLLMCommitMessageProvider } from "../liteLLMCommitProvider";
import type { LiteLLMProviderRegistry } from "../liteLLMProviderRegistry";
import type { LiteLLMModelInfo, OpenAIChatCompletionRequest } from "../../types";

const rawId = "chatgpt/gpt-6.1-sol";
const fullId = `proxy.example/${rawId}`;
const model = Object.freeze(createMockModel({ id: fullId, maxInputTokens: 1000, maxOutputTokens: 128 }));
const modelInfo: LiteLLMModelInfo = Object.freeze({ max_input_tokens: 1000, max_output_tokens: 128 });
const options = { modelOptions: {}, tools: [] } as unknown as vscode.ProvideLanguageModelChatResponseOptions;
const inputBudget = 950;

interface ProviderAccess {
    _registry: LiteLLMProviderRegistry;
    getCallTimeConfiguration(): Promise<Record<string, unknown>>;
    buildOpenAIChatRequest: RequestBuilder["buildOpenAIChatRequest"];
    sendRequestToLiteLLM(request: OpenAIChatCompletionRequest): Promise<ReadableStream<Uint8Array>>;
    sendOnceWithOverflow(
        request: OpenAIChatCompletionRequest,
        messages: readonly vscode.LanguageModelChatRequestMessage[],
        host: vscode.LanguageModelChatInformation,
        requestOptions: vscode.ProvideLanguageModelChatResponseOptions,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        token: vscode.CancellationToken,
        caller?: string,
        info?: LiteLLMModelInfo
    ): Promise<ReadableStream<Uint8Array>>;
}

function history(): vscode.LanguageModelChatRequestMessage[] {
    return [
        vscode.LanguageModelChatMessage.User("a ".repeat(200)),
        vscode.LanguageModelChatMessage.Assistant("a ".repeat(200)),
        vscode.LanguageModelChatMessage.User("a ".repeat(200)),
    ];
}

function learnCalibration(): void {
    recordCalibrationObservation(rawId, 4000, 1000);
    recordCalibrationObservation(rawId, 4000, 1000);
}

suite("Request trimming uses the usage calibration key", () => {
    let sandbox: sinon.SinonSandbox;
    let builder: RequestBuilder;
    let registries: LiteLLMProviderRegistry[];

    setup(() => {
        resetCalibration();
        sandbox = sinon.createSandbox();
        registries = [];
        const configManager = sandbox.createStubInstance(ConfigManager);
        configManager.getConfig.resolves({});
        builder = new RequestBuilder({
            configManager,
            getReasoningEffort: () => undefined,
            detectQuotaToolRedaction: (_messages, tools) => ({ tools, confidence: "none" }),
            stripUnsupportedParametersFromRequest: noop,
            isParameterSupported: () => true,
            getTelemetryOptions: () => ({}),
            usageOptOutModels: new Set(),
            extractRawModelName: (id) => (id === fullId ? rawId : id),
        });
    });

    teardown(() => {
        for (const registry of registries) {
            registry.dispose();
        }
        sandbox.restore();
        resetCalibration();
    });

    test("trims namespaced history using calibration learned from upstream usage", async () => {
        const messages = history();
        const initial = await builder.buildOpenAIChatRequest(messages, model, options, modelInfo);
        assert.strictEqual(initial.messages.length, 3);
        assert.strictEqual(tokenUtils.countOpenAIChatMessagesTokens(initial.messages, rawId, modelInfo), 780);

        const capture = new StreamTokenCapture(rawId, { report: noop }, modelInfo);
        capture.setEstimatedPromptTokens(780);
        capture.progress.report(
            new vscode.LanguageModelDataPart(
                new TextEncoder().encode(JSON.stringify({ prompt_tokens: 1560, completion_tokens: 1 })),
                "usage"
            )
        );
        assert.strictEqual(capture.getSnapshot().sawUpstreamUsage, true);
        assert.strictEqual(getCalibrationFactor(rawId), 1.3);
        assert.strictEqual(getCalibrationFactor(fullId), 1);
        const trim = sandbox.spy(tokenUtils, "trimMessagesToFitBudget");

        const request = await builder.buildOpenAIChatRequest(messages, model, options, modelInfo);

        assert.strictEqual(request.model, rawId);
        assert.strictEqual(request.messages.length, 2);
        assert.strictEqual(request.messages[0].content, "a ".repeat(200));
        assert.strictEqual(request.messages[1].role, "user");
        assert.strictEqual(request.messages[1].content, "a ".repeat(200));
        assert.strictEqual(tokenUtils.countOpenAIChatMessagesTokens(request.messages, rawId, modelInfo), 676);
        assert.strictEqual(trim.firstCall.args[2], model, "trim telemetry must retain the original model identity");
        assert.strictEqual(model.maxInputTokens, 1000);
        assert.strictEqual(request.max_tokens, 128);
        assert.strictEqual(getCalibrationFactor(fullId), 1);
    });

    test("accepts the exact calibrated input budget and rejects the next larger message", async () => {
        learnCalibration();
        const fitting = await builder.buildOpenAIChatRequest(
            [vscode.LanguageModelChatMessage.User("a ".repeat(365))],
            model,
            options,
            modelInfo
        );
        assert.strictEqual(tokenUtils.countOpenAIChatMessagesTokens(fitting.messages, rawId, modelInfo), inputBudget);
        assert.strictEqual(tokenUtils.countTokens("a ".repeat(366), rawId, modelInfo), 952);
        await assert.rejects(
            () =>
                builder.buildOpenAIChatRequest(
                    [vscode.LanguageModelChatMessage.User("a ".repeat(366))],
                    model,
                    options,
                    modelInfo
                ),
            /Message exceeds token limit\./
        );
    });

    test("preserves a calibrated system prompt and the newest message", async () => {
        learnCalibration();
        const system: vscode.LanguageModelChatRequestMessage = {
            role: 3 as vscode.LanguageModelChatMessageRole,
            content: [new vscode.LanguageModelTextPart("a ".repeat(200))],
            name: undefined,
        };
        // System 520 + newest 260 fits 950; an uncalibrated system cost of 260
        // would incorrectly admit another 260-token history turn before the wire guard.
        const messages = [
            system,
            vscode.LanguageModelChatMessage.User("a ".repeat(100)),
            vscode.LanguageModelChatMessage.Assistant("a ".repeat(100)),
            vscode.LanguageModelChatMessage.User("a ".repeat(100)),
        ];
        const request = await builder.buildOpenAIChatRequest(messages, model, options, modelInfo);
        assert.deepStrictEqual(
            request.messages.map((part) => part.role),
            ["system", "user"]
        );
        assert.strictEqual(request.messages[0].content, "a ".repeat(200));
        assert.strictEqual(tokenUtils.countOpenAIChatMessagesTokens(request.messages, rawId, modelInfo), 780);
    });

    for (const Provider of [LiteLLMChatProvider, LiteLLMCommitMessageProvider]) {
        test(`${Provider.name} calibrates the hard-budget overflow trim before rebuilding and retrying`, async () => {
            sandbox.stub(ConfigManager.prototype, "getConfig").resolves({});
            const access = new Provider(createMockSecrets(), "calibration-test") as unknown as ProviderAccess;
            registries.push(access._registry);
            learnCalibration();
            sandbox.stub(access, "getCallTimeConfiguration").resolves({});
            const rebuild = sandbox.spy(access, "buildOpenAIChatRequest");
            const send = sandbox.stub(access, "sendRequestToLiteLLM");
            send.onFirstCall().rejects(new Error("context length exceeded"));
            const stream = new ReadableStream<Uint8Array>({ start: (controller) => controller.close() });
            send.onSecondCall().resolves(stream);
            const messages = history();
            const source = new vscode.CancellationTokenSource();
            try {
                const result = await access.sendOnceWithOverflow(
                    { model: rawId, messages: [{ role: "user", content: "recent" }], max_tokens: 128 },
                    messages,
                    model,
                    options,
                    { report: noop },
                    source.token,
                    "calibration-test",
                    modelInfo
                );
                assert.strictEqual(result, stream);
                sinon.assert.calledTwice(send);
                assert.deepStrictEqual(rebuild.firstCall.args[0], [messages[2]]);
                assert.strictEqual(rebuild.firstCall.args[1], model);
                const retry = send.secondCall.args[0];
                assert.strictEqual(retry.messages.length, 1);
                assert.ok(tokenUtils.countOpenAIChatMessagesTokens(retry.messages, rawId, modelInfo) <= inputBudget);
            } finally {
                source.dispose();
            }
        });
    }
});
