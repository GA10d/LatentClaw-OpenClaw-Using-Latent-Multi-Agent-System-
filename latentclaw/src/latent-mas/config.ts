import type { Api, Model } from "@mariozechner/pi-ai";
import type { ThinkLevel } from "../auto-reply/thinking.js";
import type { OpenClawConfig } from "../config/config.js";
import type { ModelCompatConfig } from "../config/types.models.js";
import type { AgentStreamParams } from "../commands/agent/types.js";
import { normalizeProviderId } from "../agents/model-selection.js";
import { resolveModel } from "../agents/pi-embedded-runner/model.js";

export type LatentMasCompatConfig = NonNullable<ModelCompatConfig["latentMas"]>;
export type LatentMasPromptMode = NonNullable<LatentMasCompatConfig["prompt"]>;
export type LatentMasTorchDtype = NonNullable<LatentMasCompatConfig["torchDtype"]>;

export type ResolvedLatentMasConfig = {
  hfModelName: string;
  pythonBin?: string;
  latentSteps: number;
  prompt: LatentMasPromptMode;
  maxNewTokens: number;
  temperature: number;
  topP: number;
  device?: string;
  trustRemoteCode: boolean;
  thinkToken: boolean;
  latentSpaceRealign: boolean;
  torchDtype: LatentMasTorchDtype;
  historyTurns: number;
};

const LATENTMAS_PROVIDER = "latentmas";
const DEFAULT_MAX_NEW_TOKENS = 1024;
const DEFAULT_TEMPERATURE = 0.6;
const DEFAULT_TOP_P = 0.95;
const DEFAULT_HISTORY_TURNS = 6;
const DEFAULT_PROMPT: LatentMasPromptMode = "sequential";
const DEFAULT_TORCH_DTYPE: LatentMasTorchDtype = "auto";

const THINK_LEVEL_TO_STEPS: Record<ThinkLevel, number> = {
  off: 4,
  minimal: 6,
  low: 8,
  medium: 12,
  high: 16,
  xhigh: 24,
  adaptive: 12,
};

function clampInteger(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.floor(value)));
}

function normalizePositiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

function normalizeProbability(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 && value <= 1
    ? value
    : undefined;
}

function normalizeNonNegativeNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function readLatentMasCompat(compat: unknown): LatentMasCompatConfig | undefined {
  if (!compat || typeof compat !== "object" || Array.isArray(compat)) {
    return undefined;
  }
  const value = (compat as { latentMas?: unknown }).latentMas;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as LatentMasCompatConfig;
}

function resolveDefaultLatentSteps(thinkLevel?: ThinkLevel): number {
  if (!thinkLevel) {
    return THINK_LEVEL_TO_STEPS.adaptive;
  }
  return THINK_LEVEL_TO_STEPS[thinkLevel];
}

export function resolveLatentMasConfig(params: {
  provider: string;
  modelId: string;
  cfg: OpenClawConfig;
  agentDir: string;
  thinkLevel?: ThinkLevel;
  streamParams?: AgentStreamParams;
}): ResolvedLatentMasConfig | null {
  const provider = normalizeProviderId(params.provider);
  const resolved = resolveModel(provider, params.modelId, params.agentDir, params.cfg).model;
  const compat = readLatentMasCompat((resolved as Model<Api> | undefined)?.compat);
  const enabled = compat?.enabled ?? provider === LATENTMAS_PROVIDER;
  if (!enabled) {
    return null;
  }

  const streamMaxTokens = normalizePositiveInteger(params.streamParams?.maxTokens);
  const streamTemperature = normalizeNonNegativeNumber(params.streamParams?.temperature);
  const compatMaxNewTokens = normalizePositiveInteger(compat?.maxNewTokens);
  const compatTemperature = normalizeNonNegativeNumber(compat?.temperature);
  const compatTopP = normalizeProbability(compat?.topP);
  const compatLatentSteps =
    typeof compat?.latentSteps === "number" && Number.isFinite(compat.latentSteps)
      ? clampInteger(compat.latentSteps, 0, 80)
      : undefined;
  const compatHistoryTurns = normalizePositiveInteger(compat?.historyTurns);
  const prompt = compat?.prompt === "hierarchical" ? "hierarchical" : DEFAULT_PROMPT;
  const hfModelName = compat?.modelName?.trim() || params.modelId.trim();

  return {
    hfModelName,
    pythonBin: compat?.pythonBin?.trim() || undefined,
    latentSteps: compatLatentSteps ?? resolveDefaultLatentSteps(params.thinkLevel),
    prompt,
    maxNewTokens: streamMaxTokens ?? compatMaxNewTokens ?? DEFAULT_MAX_NEW_TOKENS,
    temperature: streamTemperature ?? compatTemperature ?? DEFAULT_TEMPERATURE,
    topP: compatTopP ?? DEFAULT_TOP_P,
    device: compat?.device?.trim() || undefined,
    trustRemoteCode: compat?.trustRemoteCode === true,
    thinkToken: compat?.thinkToken === true,
    latentSpaceRealign: compat?.latentSpaceRealign !== false,
    torchDtype: compat?.torchDtype ?? DEFAULT_TORCH_DTYPE,
    historyTurns: clampInteger(compatHistoryTurns ?? DEFAULT_HISTORY_TURNS, 1, 32),
  };
}
