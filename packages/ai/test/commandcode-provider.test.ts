import { describe, expect, it } from "vitest";
import { commandCodeGenerateApi } from "../src/api/commandcode-generate.lazy.ts";
import { builtinModels, builtinProviders } from "../src/providers/all.ts";
import { COMMAND_CODE_MODELS, seedForUnknownModel } from "../src/providers/commandcode.catalog.ts";
import {
	COMMAND_CODE_PROVIDER_ID,
	commandCodeModelsFromApiResponse,
	commandCodeProvider,
	commandCodeTransport,
	resetCommandCodeTransport,
	staticCommandCodeModels,
} from "../src/providers/commandcode.ts";
import {
	commandCodeErrorMessage,
	isCommandCodeUpgradeRequired,
	readableCommandCodeError,
} from "../src/providers/commandcode-errors.ts";
import type { Context } from "../src/types.ts";

const context: Context = { messages: [{ role: "user", content: "hi", timestamp: Date.now() }] };

function upgradeRequiredResponse(): Response {
	return new Response(
		JSON.stringify({
			error: {
				message: "Your Go plan doesn't include API access.",
				type: "permission_error",
				code: "upgrade_required",
			},
		}),
		{ status: 403, headers: { "content-type": "application/json" } },
	);
}

function sseResponse(events: unknown[]): Response {
	const body = `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`;
	return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

describe("commandcode catalog", () => {
	it("registers the provider with every static model", () => {
		const models = builtinModels();
		const list = models.getModels(COMMAND_CODE_PROVIDER_ID);
		expect(builtinProviders().map((provider) => provider.id)).toContain(COMMAND_CODE_PROVIDER_ID);
		expect(list.length).toBe(COMMAND_CODE_MODELS.length);
		expect(list.every((model) => model.provider === COMMAND_CODE_PROVIDER_ID)).toBe(true);
	});

	it("routes claude models to anthropic-messages and everything else to openai-completions", () => {
		const models = builtinModels();
		expect(models.getModel(COMMAND_CODE_PROVIDER_ID, "claude-sonnet-5")?.api).toBe("anthropic-messages");
		expect(models.getModel(COMMAND_CODE_PROVIDER_ID, "deepseek/deepseek-v4.1-flash")?.api).toBe("openai-completions");
	});

	it("carries vision, reasoning, and pricing metadata", () => {
		const models = builtinModels();
		const sonnet = models.getModel(COMMAND_CODE_PROVIDER_ID, "claude-sonnet-5");
		expect(sonnet?.input).toContain("image");
		expect(sonnet?.reasoning).toBe(true);
		expect(sonnet?.cost).toEqual({ input: 2, output: 10, cacheRead: 0.2, cacheWrite: 2.5 });
		expect(sonnet?.thinkingLevelMap?.high).toBe("high");
		// Claude models do not accept the minimal effort.
		expect(sonnet?.thinkingLevelMap?.minimal).toBeNull();
	});

	it("defaults unknown models to text-only with zero cost", () => {
		const seed = seedForUnknownModel("brand/new-model", "brand/new", 200_000);
		expect(seed.api).toBe("openai-completions");
		expect(seed.reasoning).toBe(false);
		expect(seed.input).toEqual(["text"]);
		expect(seed.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
	});

	it("merges the live catalog over static capabilities", () => {
		const parsed = commandCodeModelsFromApiResponse({
			object: "list",
			data: [
				{ id: "claude-sonnet-5", name: "Claude Sonnet 5", context_length: 500_000 },
				{ id: "brand/new-model", name: "Brand New", context_length: 123_456 },
			],
		});
		expect(parsed).toHaveLength(2);
		// Static capability metadata survives the live context_length override.
		expect(parsed[0]).toMatchObject({ id: "claude-sonnet-5", contextWindow: 500_000, reasoning: true });
		expect(parsed[1]).toMatchObject({ id: "brand/new-model", contextWindow: 123_456, reasoning: false });
	});
});

describe("commandcode transport fallback", () => {
	it("falls back to /alpha/generate when the Provider API reports upgrade_required", async () => {
		resetCommandCodeTransport();
		const requested: string[] = [];
		const fetchImpl = (async (input: Parameters<typeof globalThis.fetch>[0]) => {
			const url = String(input);
			requested.push(url);
			if (url.includes("/provider/v1/")) return upgradeRequiredResponse();
			return sseResponse([
				{ type: "start" },
				{ type: "text-start", id: "txt-0" },
				{ type: "text-delta", id: "txt-0", text: "hello" },
				{ type: "text-end", id: "txt-0" },
				{
					type: "finish",
					finishReason: "stop",
					rawFinishReason: "stop",
					totalUsage: { inputTokens: 10, outputTokens: 1 },
				},
			]);
		}) as typeof fetch;

		const provider = commandCodeProvider({ fetchImpl });
		const model = provider.getModels().find((entry) => entry.id === "deepseek/deepseek-v4.1-flash")!;
		const events: string[] = [];
		const stream = provider.streamSimple(model, context, { apiKey: "test-key", fetch: fetchImpl });
		for await (const event of stream) events.push(event.type);

		expect(requested[0]).toContain("/provider/v1/");
		expect(requested[1]).toContain("/alpha/generate");
		expect(events).toEqual(["start", "text_start", "text_delta", "text_end", "done"]);
		const result = await stream.result();
		expect(result.stopReason).toBe("stop");
		expect(result.content).toEqual([{ type: "text", text: "hello" }]);
		// The transport pin skips the doomed provider round trip afterwards.
		expect(commandCodeTransport("test-key")).toBe("generate");
		resetCommandCodeTransport();
	});

	it("parses reasoning, tool calls, and usage from the generate stream", async () => {
		resetCommandCodeTransport();
		const fetchImpl = (async (input: Parameters<typeof globalThis.fetch>[0]) => {
			if (String(input).includes("/provider/v1/")) return upgradeRequiredResponse();
			return sseResponse([
				{ type: "reasoning-start", id: "r-0" },
				{ type: "reasoning-delta", id: "r-0", text: "thinking" },
				{ type: "reasoning-end", id: "r-0" },
				{ type: "tool-input-start", id: "call-1", toolName: "ls" },
				{ type: "tool-input-delta", id: "call-1", delta: '{"path":' },
				{ type: "tool-input-delta", id: "call-1", delta: '"."}' },
				{ type: "tool-call", toolCallId: "call-1", toolName: "ls", input: { path: "." } },
				{
					type: "finish",
					finishReason: "tool-calls",
					rawFinishReason: "tool-calls",
					totalUsage: {
						inputTokens: 100,
						inputTokenDetails: { noCacheTokens: 40, cacheReadTokens: 60 },
						outputTokens: 7,
					},
				},
			]);
		}) as typeof fetch;

		const provider = commandCodeProvider({ fetchImpl });
		const model = provider.getModels().find((entry) => entry.id === "deepseek/deepseek-v4.1-flash")!;
		const stream = provider.streamSimple(model, context, { apiKey: "test-key", fetch: fetchImpl });
		const result = await stream.result();

		expect(result.stopReason).toBe("toolUse");
		expect(result.content[0]).toEqual({ type: "thinking", thinking: "thinking" });
		expect(result.content[1]).toMatchObject({ type: "toolCall", id: "call-1", name: "ls", arguments: { path: "." } });
		expect(result.usage).toMatchObject({ input: 40, cacheRead: 60, output: 7, totalTokens: 107 });
		resetCommandCodeTransport();
	});

	it("uses the provider API directly when the account has access", async () => {
		resetCommandCodeTransport();
		const requested: string[] = [];
		const fetchImpl = (async (input: Parameters<typeof globalThis.fetch>[0]) => {
			requested.push(String(input));
			return sseResponse([
				{ id: "chatcmpl-1", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "hi" } }] },
				{
					id: "chatcmpl-1",
					object: "chat.completion.chunk",
					choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
					usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
				},
			]);
		}) as typeof fetch;

		const provider = commandCodeProvider({ fetchImpl });
		const model = provider.getModels().find((entry) => entry.id === "deepseek/deepseek-v4.1-flash")!;
		const result = await provider.streamSimple(model, context, { apiKey: "test-key", fetch: fetchImpl }).result();

		expect(requested).toHaveLength(1);
		expect(requested[0]).toContain("/provider/v1/chat/completions");
		expect(result.stopReason).toBe("stop");
		expect(commandCodeTransport("test-key")).toBe("unknown");
		resetCommandCodeTransport();
	});
});

describe("commandcode api registration", () => {
	it("exposes a lazy generate API", () => {
		expect(typeof commandCodeGenerateApi().stream).toBe("function");
		expect(typeof commandCodeGenerateApi().streamSimple).toBe("function");
	});

	it("exposes the static catalog for offline use", () => {
		const staticModels = staticCommandCodeModels();
		expect(staticModels.length).toBe(COMMAND_CODE_MODELS.length);
		expect(staticModels[0].id).toBe(COMMAND_CODE_MODELS[0].id);
	});
});

describe("commandcode error classification", () => {
	it("detects the OpenAI-shaped upgrade_required body", () => {
		expect(
			isCommandCodeUpgradeRequired(403, {
				error: {
					message: "Your Go plan doesn't include API access.",
					type: "permission_error",
					code: "upgrade_required",
				},
			}),
		).toBe(true);
	});

	it("detects the Anthropic-shaped permission_error body with no code", () => {
		expect(
			isCommandCodeUpgradeRequired(403, {
				type: "error",
				error: {
					type: "permission_error",
					message: "Your Go plan doesn't include API access. Upgrade to Provider or higher.",
				},
			}),
		).toBe(true);
	});

	it("does not treat a per-model plan restriction as an upgrade", () => {
		expect(
			isCommandCodeUpgradeRequired(403, {
				success: false,
				error: {
					code: "FORBIDDEN",
					status: 403,
					message: "MODEL_NOT_IN_PLAN: Claude Sonnet 5 available in Pro and above plans",
				},
			}),
		).toBe(false);
	});

	it("ignores non-403 statuses", () => {
		expect(isCommandCodeUpgradeRequired(500, { error: { code: "upgrade_required" } })).toBe(false);
		expect(isCommandCodeUpgradeRequired(401, { error: { type: "permission_error", message: "plan" } })).toBe(false);
	});

	it("extracts a readable message from a nested error body", () => {
		expect(
			commandCodeErrorMessage(
				JSON.stringify({
					success: false,
					error: { code: "FORBIDDEN", message: "MODEL_NOT_IN_PLAN: Claude Sonnet 5 available in Pro and above" },
				}),
			),
		).toBe("FORBIDDEN: MODEL_NOT_IN_PLAN: Claude Sonnet 5 available in Pro and above");
	});

	it("rewrites embedded JSON in a provider error message", () => {
		const raw =
			'Command Code API error 403: {"success":false,"error":{"code":"FORBIDDEN","message":"MODEL_NOT_IN_PLAN: nope"}}';
		expect(readableCommandCodeError(raw)).toBe("Command Code API error 403: FORBIDDEN: MODEL_NOT_IN_PLAN: nope");
	});
});
