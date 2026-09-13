import { FinishReason } from "@google/genai";
import { describe, expect, it } from "vitest";
import { mapStopReason, mapStopReasonString } from "../src/api/google-shared.ts";

describe("mapStopReason", () => {
	it("maps the terminal success and truncation reasons", () => {
		expect(mapStopReason(FinishReason.STOP)).toBe("stop");
		expect(mapStopReason(FinishReason.MAX_TOKENS)).toBe("length");
	});

	it("maps every non-terminal reason to error", () => {
		for (const reason of [
			FinishReason.BLOCKLIST,
			FinishReason.PROHIBITED_CONTENT,
			FinishReason.SPII,
			FinishReason.SAFETY,
			FinishReason.IMAGE_SAFETY,
			FinishReason.IMAGE_PROHIBITED_CONTENT,
			FinishReason.IMAGE_RECITATION,
			FinishReason.IMAGE_OTHER,
			FinishReason.RECITATION,
			FinishReason.FINISH_REASON_UNSPECIFIED,
			FinishReason.OTHER,
			FinishReason.LANGUAGE,
			FinishReason.MALFORMED_FUNCTION_CALL,
			FinishReason.UNEXPECTED_TOOL_CALL,
			FinishReason.TOO_MANY_TOOL_CALLS,
			FinishReason.NO_IMAGE,
		]) {
			expect(mapStopReason(reason)).toBe("error");
		}
	});

	it("handles every FinishReason member without throwing", () => {
		// The switch is exhaustive, so a newly added SDK member must be handled
		// here too rather than crashing at runtime.
		for (const reason of Object.values(FinishReason)) {
			expect(() => mapStopReason(reason)).not.toThrow();
		}
	});
});

describe("mapStopReasonString", () => {
	it("maps raw API strings and defaults to error", () => {
		expect(mapStopReasonString("STOP")).toBe("stop");
		expect(mapStopReasonString("MAX_TOKENS")).toBe("length");
		expect(mapStopReasonString("TOO_MANY_TOOL_CALLS")).toBe("error");
		expect(mapStopReasonString("SOMETHING_NEW")).toBe("error");
	});
});
