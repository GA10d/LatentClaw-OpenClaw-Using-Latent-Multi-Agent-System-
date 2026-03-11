import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { resolvePythonExecutablePath } from "../hooks/gmail-setup-utils.js";
import { runCommandWithTimeout } from "../process/exec.js";
import type { ResolvedLatentMasConfig } from "./config.js";

type LatentMasBridgePayload = {
  prompt: string;
  extraSystemPrompt?: string;
  sessionContext?: string;
  config: ResolvedLatentMasConfig;
};

type LatentMasBridgeResult = {
  text: string;
  usage?: {
    input?: number;
    output?: number;
    total?: number;
  };
};

function resolveBridgeScriptPath(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  let current = here;
  for (let depth = 0; depth < 6; depth += 1) {
    const candidate = path.join(current, "assets", "latent-mas", "bridge.py");
    if (fs.existsSync(candidate)) {
      return candidate;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  throw new Error("LatentMAS bridge script not found (expected assets/latent-mas/bridge.py).");
}

function formatBridgeFailure(stderr: string, stdout: string): string {
  const detail = stderr.trim() || stdout.trim();
  return detail ? `LatentMAS bridge failed: ${detail}` : "LatentMAS bridge failed.";
}

export async function runLatentMasBridge(params: {
  config: ResolvedLatentMasConfig;
  prompt: string;
  extraSystemPrompt?: string;
  sessionContext?: string;
  timeoutMs?: number;
  workspaceDir?: string;
}): Promise<LatentMasBridgeResult> {
  const pythonBin = params.config.pythonBin ?? (await resolvePythonExecutablePath());
  if (!pythonBin) {
    throw new Error("LatentMAS requires a local Python interpreter (python3 or python) on PATH.");
  }

  const payload: LatentMasBridgePayload = {
    prompt: params.prompt,
    extraSystemPrompt: params.extraSystemPrompt,
    sessionContext: params.sessionContext,
    config: params.config,
  };
  const timeoutMs =
    typeof params.timeoutMs === "number" && Number.isFinite(params.timeoutMs) && params.timeoutMs > 0
      ? params.timeoutMs
      : 60 * 60 * 1000;
  const bridgePath = resolveBridgeScriptPath();
  const result = await runCommandWithTimeout([pythonBin, bridgePath], {
    timeoutMs,
    cwd: params.workspaceDir,
    input: JSON.stringify(payload),
    env: {
      PYTHONUTF8: "1",
    },
  });
  if (result.code !== 0) {
    throw new Error(formatBridgeFailure(result.stderr, result.stdout));
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(result.stdout);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`LatentMAS bridge returned invalid JSON: ${message}`);
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("LatentMAS bridge returned an invalid payload.");
  }
  const text =
    typeof (parsed as { text?: unknown }).text === "string" ? (parsed as { text: string }).text : "";
  if (!text.trim()) {
    throw new Error("LatentMAS bridge returned an empty assistant response.");
  }

  const usageRaw = (parsed as { usage?: unknown }).usage;
  const usage =
    usageRaw && typeof usageRaw === "object" && !Array.isArray(usageRaw)
      ? {
          input:
            typeof (usageRaw as { input?: unknown }).input === "number"
              ? (usageRaw as { input: number }).input
              : undefined,
          output:
            typeof (usageRaw as { output?: unknown }).output === "number"
              ? (usageRaw as { output: number }).output
              : undefined,
          total:
            typeof (usageRaw as { total?: unknown }).total === "number"
              ? (usageRaw as { total: number }).total
              : undefined,
        }
      : undefined;

  return { text, usage };
}
