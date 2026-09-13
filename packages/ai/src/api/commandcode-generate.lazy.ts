import type { ProviderStreams } from "../types.ts";
import { lazyApi } from "./lazy.ts";

export const commandCodeGenerateApi = (): ProviderStreams => lazyApi(() => import("./commandcode-generate.ts"));
