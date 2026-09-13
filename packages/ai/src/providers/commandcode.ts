import { anthropicMessagesApi } from "../api/anthropic-messages.lazy.ts";
import { commandCodeGenerateApi } from "../api/commandcode-generate.lazy.ts";
import { openAICompletionsApi } from "../api/openai-completions.lazy.ts";
import type { ApiKeyAuth } from "../auth/types.ts";
import { createProvider, type Provider } from "../models.ts";
import type {
	Api,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	Context,
	Model,
	ProviderStreams,
	SimpleStreamOptions,
	StreamOptions,
} from "../types.ts";
import { AssistantMessageEventStream as EventStream } from "../utils/event-stream.ts";
import { COMMAND_CODE_MODELS, type CommandCodeModelSeed, seedForUnknownModel } from "./commandcode.catalog.ts";
import {
	COMMAND_CODE_UPGRADE_CODE,
	isCommandCodeUpgradeRequired,
	readableCommandCodeError,
} from "./commandcode-errors.ts";

export const COMMAND_CODE_PROVIDER_ID = "commandcode";
export const COMMAND_CODE_PROVIDER_API_BASE = "https://api.commandcode.ai/provider/v1";
export const COMMAND_CODE_MODELS_URL = `${COMMAND_CODE_PROVIDER_API_BASE}/models`;
export const COMMAND_CODE_API = "commandcode-generate";

const DEFAULT_MODELS_TIMEOUT_MS = 10_000;

type CommandCodeApi = "anthropic-messages" | "openai-completions" | typeof COMMAND_CODE_API;

interface CommandCodeProviderOptions {
	/** Provider API base used for both model discovery and the native adapters. */
	apiBase?: string;
	/** Override the discovery endpoint. */
	modelsUrl?: string;
	/** Discovery timeout in milliseconds. */
	timeoutMs?: number;
	/** Injectable fetch for tests and compatible endpoints. */
	fetchImpl?: typeof fetch;
}

export interface CommandCodeModel {
	id: string;
	name: string;
	api: "anthropic-messages" | "openai-completions";
	reasoning: boolean;
	thinkingLevelMap: Record<string, string | null>;
	input: ("text" | "image")[];
	cost: Model<string>["cost"];
	contextWindow: number;
	maxTokens: number;
}

/** Error code the Provider API returns when the plan excludes API access. */
export { COMMAND_CODE_UPGRADE_CODE };

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | undefined {
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

function positiveNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

/**
 * Transport state for the process. `/alpha/generate` is the fallback for plans
 * without Provider API access; once a request answers `403 upgrade_required`,
 * the transport is pinned so later requests skip a doomed round trip. The pin
 * is keyed by API key so switching accounts re-probes.
 */
let pinnedTransport: "unknown" | "generate" = "unknown";
let pinnedApiKey: string | undefined;

export function resetCommandCodeTransport(): void {
	pinnedTransport = "unknown";
	pinnedApiKey = undefined;
}

export function commandCodeTransport(apiKey: string | undefined): "unknown" | "generate" {
	if (apiKey !== pinnedApiKey) {
		pinnedApiKey = apiKey;
		pinnedTransport = "unknown";
	}
	return pinnedTransport;
}

function pinGenerateTransport(apiKey: string | undefined): void {
	if (apiKey === pinnedApiKey) pinnedTransport = "generate";
}

/**
 * Detect the Provider API's plan restriction. The body shape differs per
 * compatible surface, so classification lives in `commandcode-errors.ts`.
 */
async function isUpgradeRequired(response: Response): Promise<boolean> {
	if (response.status !== 403) return false;
	try {
		return isCommandCodeUpgradeRequired(response.status, await response.clone().json());
	} catch {
		return false;
	}
}

function errorEvent(model: Model<Api>, error: unknown, aborted: boolean): AssistantMessageEvent {
	const reason = aborted ? "aborted" : "error";
	const raw = error instanceof Error ? error.message : String(error);
	return {
		type: "error",
		reason,
		error: {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: reason,
			errorMessage: readableCommandCodeError(raw),
			timestamp: Date.now(),
		},
	};
}

/**
 * Run the native Provider API adapter, and if the plan restriction surfaces,
 * reissue the request through `/alpha/generate`. Events emitted by the native
 * adapter before the restriction is detected are withheld so the caller sees
 * exactly one coherent response.
 */
function withUpgradeFallback(
	model: Model<Api>,
	context: Context,
	options: StreamOptions | SimpleStreamOptions | undefined,
	simple: boolean,
	startNative: (options: StreamOptions | SimpleStreamOptions) => AssistantMessageEventStream,
	generate: ProviderStreams,
): AssistantMessageEventStream {
	const output = new EventStream();
	const apiKey = options?.apiKey;
	let upgradeRequired = false;

	const callerFetch = options?.fetch;
	const probeFetch = async (input: Parameters<typeof globalThis.fetch>[0], init?: RequestInit): Promise<Response> => {
		const response = await (callerFetch ?? globalThis.fetch)(input, init);
		if (await isUpgradeRequired(response)) upgradeRequired = true;
		return response;
	};
	const probeOptions = {
		...options,
		fetch: probeFetch,
		onResponse: upgradeRequired ? undefined : options?.onResponse,
	} as StreamOptions | SimpleStreamOptions;

	void (async () => {
		try {
			for await (const event of startNative(probeOptions)) {
				if (!upgradeRequired) output.push(event);
			}
		} catch (error) {
			if (!upgradeRequired) {
				output.push(errorEvent(model, error, options?.signal?.aborted ?? false));
				output.end();
				return;
			}
		}

		if (!upgradeRequired) {
			output.end();
			return;
		}

		pinGenerateTransport(apiKey);
		const fallbackOptions = { ...options, fetch: callerFetch } as SimpleStreamOptions;
		const fallback = simple
			? generate.streamSimple(model, context, fallbackOptions)
			: generate.stream(model, context, fallbackOptions);
		try {
			for await (const event of fallback) output.push(event);
			output.end(await fallback.result());
		} catch (error) {
			output.push(errorEvent(model, error, options?.signal?.aborted ?? false));
			output.end();
		}
	})();

	return output;
}

function buildModel(seed: CommandCodeModelSeed, apiBase: string): Model<Api> {
	const model: Model<Api> = {
		id: seed.id,
		name: seed.name,
		api: seed.api,
		provider: COMMAND_CODE_PROVIDER_ID,
		baseUrl: apiBase,
		reasoning: seed.reasoning,
		thinkingLevelMap: seed.thinkingLevelMap as Model<Api>["thinkingLevelMap"],
		input: [...seed.input],
		cost: seed.cost,
		contextWindow: seed.contextWindow,
		maxTokens: seed.maxTokens,
	};
	if (seed.api === "openai-completions") {
		model.compat = {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: seed.reasoning,
			supportsUsageInStreaming: true,
			maxTokensField: "max_tokens",
		};
	} else {
		model.compat = {
			supportsEagerToolInputStreaming: false,
			supportsLongCacheRetention: false,
			supportsCacheControlOnTools: false,
			supportsToolReferences: false,
			...(seed.reasoning ? { forceAdaptiveThinking: true } : {}),
		};
	}
	return model;
}

/**
 * Command Code API key: environment first, matching the official CLI. The
 * credential store is consulted by the host before ambient resolution.
 */
export function commandCodeAuth(): ApiKeyAuth {
	return {
		name: "Command Code API key",
		login: async (interaction) => {
			interaction.signal.throwIfAborted();
			const method = await interaction.prompt({
				type: "select",
				message: "Command Code authentication:",
				options: [
					{ id: "browser", label: "Browser login", description: "Create a key in Command Code Studio" },
					{ id: "key", label: "Paste an API key" },
				],
			});
			interaction.signal.throwIfAborted();
			if (method === "browser") {
				interaction.notify({
					type: "info",
					message: "Copy an API key from Command Code Studio, then paste it here.",
					links: [{ label: "Command Code Studio", url: "https://commandcode.ai" }],
				});
			}
			const key = await interaction.prompt({ type: "secret", message: "Enter Command Code API key" });
			interaction.signal.throwIfAborted();
			return { type: "api_key", key: key.trim() };
		},
		resolve: async ({ ctx, credential, signal }) => {
			signal.throwIfAborted();
			if (credential?.key) return { auth: { apiKey: credential.key }, source: "stored credential" };
			for (const name of ["COMMAND_CODE_API_KEY", "COMMANDCODE_API_KEY"]) {
				const value = await ctx.env(name);
				signal.throwIfAborted();
				if (value) return { auth: { apiKey: value }, source: name };
			}
			return undefined;
		},
	};
}

function parseModelEntry(value: unknown): { id: string; name: string; contextWindow: number } | undefined {
	if (!isRecord(value)) return undefined;
	const id = stringValue(value.id);
	const contextLength = positiveNumber(value.context_length);
	if (!id || contextLength === undefined) return undefined;
	return { id, name: stringValue(value.name) ?? id, contextWindow: contextLength };
}

/** Merge the live catalog over the static capability and pricing table. */
export function commandCodeModelsFromApiResponse(value: unknown): CommandCodeModel[] {
	if (!isRecord(value)) throw new Error("Expected Command Code models response to be an object");
	if (!Array.isArray(value.data)) throw new Error("Expected Command Code models response data to be an array");

	const known = new Map(COMMAND_CODE_MODELS.map((seed) => [seed.id, seed]));
	return value.data.flatMap((entry) => {
		const parsed = parseModelEntry(entry);
		if (!parsed) return [];
		const seed = known.get(parsed.id) ?? seedForUnknownModel(parsed.id, parsed.name, parsed.contextWindow);
		return [{ ...seed, contextWindow: parsed.contextWindow }];
	});
}

async function fetchCommandCodeModels(
	url: string,
	fetchImpl: typeof fetch,
	timeoutMs: number,
	signal: AbortSignal,
): Promise<CommandCodeModel[]> {
	const response = await fetchImpl(url, {
		headers: { accept: "application/json" },
		signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]),
	});
	if (!response.ok) {
		throw new Error(`Failed to fetch Command Code models: ${response.status} ${response.statusText}`);
	}
	return commandCodeModelsFromApiResponse(await response.json());
}

/** Static catalog used when discovery is unavailable. */
export function staticCommandCodeModels(): CommandCodeModel[] {
	return COMMAND_CODE_MODELS.map((seed) => ({ ...seed }));
}

/**
 * Command Code provider.
 *
 * Models are registered against the provider's native OpenAI- and
 * Anthropic-compatible endpoints. Accounts whose plan excludes Provider API
 * access answer `403 upgrade_required` there, so the provider falls back to the
 * CLI's `/alpha/generate` transport for those accounts and pins it for the rest
 * of the process.
 */
export function commandCodeProvider(options: CommandCodeProviderOptions = {}): Provider<CommandCodeApi> {
	const apiBase = options.apiBase ?? process.env.COMMANDCODE_API_BASE ?? COMMAND_CODE_PROVIDER_API_BASE;
	const modelsUrl = options.modelsUrl ?? process.env.COMMANDCODE_MODELS_URL ?? COMMAND_CODE_MODELS_URL;
	const timeoutMs = options.timeoutMs ?? DEFAULT_MODELS_TIMEOUT_MS;
	const fetchImpl = options.fetchImpl ?? globalThis.fetch;

	let liveModels: CommandCodeModel[] | undefined;
	const currentModels = (): Model<CommandCodeApi>[] =>
		(liveModels ?? staticCommandCodeModels()).map((seed) => buildModel(seed, apiBase) as Model<CommandCodeApi>);

	const nativeAnthropic = anthropicMessagesApi();
	const nativeOpenAI = openAICompletionsApi();
	const generate = commandCodeGenerateApi();

	const dispatch = (
		model: Model<Api>,
		context: Context,
		options: StreamOptions | SimpleStreamOptions | undefined,
		simple: boolean,
	): AssistantMessageEventStream => {
		if (commandCodeTransport(options?.apiKey) === "generate") {
			return simple
				? generate.streamSimple(model, context, options as SimpleStreamOptions)
				: generate.stream(model, context, options);
		}
		const native = model.api === "anthropic-messages" ? nativeAnthropic : nativeOpenAI;
		const startNative = (nativeOptions: StreamOptions | SimpleStreamOptions) =>
			simple
				? native.streamSimple(model, context, nativeOptions as SimpleStreamOptions)
				: native.stream(model, context, nativeOptions);
		return withUpgradeFallback(model, context, options, simple, startNative, generate);
	};

	return createProvider<CommandCodeApi>({
		id: COMMAND_CODE_PROVIDER_ID,
		name: "Command Code",
		baseUrl: apiBase,
		auth: { apiKey: commandCodeAuth() },
		models: currentModels(),
		fetchModels: async (context): Promise<Model<CommandCodeApi>[]> => {
			try {
				liveModels = await fetchCommandCodeModels(modelsUrl, fetchImpl, timeoutMs, context.signal);
			} catch (error) {
				if (context.signal.aborted) throw error;
				// Discovery is best-effort; the static catalog stays authoritative.
				return staticCommandCodeModels().map((seed) => buildModel(seed, apiBase) as Model<CommandCodeApi>);
			}
			return currentModels();
		},
		api: {
			"anthropic-messages": {
				stream: (model, context, options) => dispatch(model, context, options, false),
				streamSimple: (model, context, options) => dispatch(model, context, options, true),
			},
			"openai-completions": {
				stream: (model, context, options) => dispatch(model, context, options, false),
				streamSimple: (model, context, options) => dispatch(model, context, options, true),
			},
			[COMMAND_CODE_API]: generate,
		} as Partial<Record<CommandCodeApi, ProviderStreams>>,
	});
}
