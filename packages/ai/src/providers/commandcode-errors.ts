/**
 * Command Code error classification shared by the provider and its transports.
 *
 * Two different 403s reach this code and they must not be confused:
 *
 * - `upgrade_required` / `permission_error` — the account's plan excludes the
 *   Provider API pair entirely, so the request belongs on `/alpha/generate`.
 * - `FORBIDDEN` with `MODEL_NOT_IN_PLAN` — the account can use the CLI but this
 *   particular model is not included. Retrying elsewhere cannot help.
 */

/** Error code the Provider API returns when the plan excludes API access. */
export const COMMAND_CODE_UPGRADE_CODE = "upgrade_required";

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | undefined {
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

/** Unwrap `{error:{...}}` or `{success:false,error:{...}}` to the error object. */
function errorObject(body: unknown): Record<string, unknown> | undefined {
	if (!isRecord(body)) return undefined;
	if (isRecord(body.error)) return body.error;
	return body;
}

/**
 * Whether a 403 means "this plan cannot use the Provider API at all".
 *
 * OpenAI-compatible endpoints answer `{error:{code:"upgrade_required"}}`;
 * Anthropic-compatible endpoints answer with no code at all, only
 * `{error:{type:"permission_error"}}` plus plan wording in the message.
 */
export function isCommandCodeUpgradeRequired(status: number, body: unknown): boolean {
	if (status !== 403) return false;
	const error = errorObject(body);
	if (!error) return false;
	if (error.code === COMMAND_CODE_UPGRADE_CODE) return true;
	if (error.type !== "permission_error") return false;
	return /plan|upgrade|subscription/i.test(stringValue(error.message) ?? "");
}

/**
 * Pull a human-readable message out of a Command Code error body so callers do
 * not surface raw JSON. Returns undefined when nothing useful is present.
 */
export function commandCodeErrorMessage(body: unknown): string | undefined {
	if (typeof body === "string") {
		const trimmed = body.trim();
		if (!trimmed) return undefined;
		try {
			return commandCodeErrorMessage(JSON.parse(trimmed)) ?? undefined;
		} catch {
			return trimmed;
		}
	}
	const error = errorObject(body);
	if (!error) return undefined;

	const message = stringValue(error.message);
	const code = stringValue(error.code);
	// `MODEL_NOT_IN_PLAN: <detail>` already leads with the code; avoid doubling it.
	if (message && code && !message.startsWith(code)) return `${code}: ${message}`;
	return message ?? code;
}

/**
 * Rewrite a provider/SDK error message that embeds a Command Code JSON body
 * into the extracted message. Non-JSON and unrelated messages pass through.
 */
export function readableCommandCodeError(message: string): string {
	const jsonStart = message.indexOf("{");
	if (jsonStart < 0) return message;
	const prefix = message.slice(0, jsonStart);
	if (!/command code|403|401|400/i.test(prefix)) return message;
	const extracted = commandCodeErrorMessage(message.slice(jsonStart));
	return extracted ? `${prefix.trimEnd()} ${extracted}`.trim() : message;
}
