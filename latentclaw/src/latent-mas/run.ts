import fs from "node:fs/promises";
import { SessionManager } from "@mariozechner/pi-coding-agent";
import type { ThinkLevel } from "../auto-reply/thinking.js";
import type { OpenClawConfig } from "../config/config.js";
import type { AgentStreamParams } from "../commands/agent/types.js";
import { prepareSessionManagerForRun } from "../agents/pi-embedded-runner/session-manager-init.js";
import type { EmbeddedPiRunResult } from "../agents/pi-embedded-runner/types.js";
import { emitSessionTranscriptUpdate } from "../sessions/transcript-events.js";
import { resolveLatentMasConfig, type ResolvedLatentMasConfig } from "./config.js";
import { runLatentMasBridge } from "./bridge.js";

type SessionHeaderEntry = { type: "session"; id?: string; cwd?: string };
type SessionMessageRecord = {
  type: "message";
  message?: {
    role?: string;
    content?: unknown;
  };
};

function extractContentText(content: unknown): string {
  if (typeof content === "string") {
    return content.trim();
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const lines = content
    .map((block) => {
      if (typeof block === "string") {
        return block.trim();
      }
      if (!block || typeof block !== "object") {
        return "";
      }
      const type = (block as { type?: unknown }).type;
      if (type === "text" && typeof (block as { text?: unknown }).text === "string") {
        return (block as { text: string }).text.trim();
      }
      if (type === "tool_use" && typeof (block as { name?: unknown }).name === "string") {
        return `[tool:${(block as { name: string }).name}]`;
      }
      if (typeof (block as { text?: unknown }).text === "string") {
        return (block as { text: string }).text.trim();
      }
      return "";
    })
    .filter(Boolean);
  return lines.join("\n").trim();
}

async function buildSessionContext(sessionFile: string, historyTurns: number): Promise<string> {
  try {
    const exists = await fs
      .access(sessionFile)
      .then(() => true)
      .catch(() => false);
    if (!exists) {
      return "";
    }

    const sessionManager = SessionManager.open(sessionFile) as {
      fileEntries?: Array<SessionHeaderEntry | SessionMessageRecord | { type?: string }>;
    };
    const entries = Array.isArray(sessionManager.fileEntries) ? sessionManager.fileEntries : [];
    const messages = entries
      .filter((entry): entry is SessionMessageRecord => entry?.type === "message")
      .map((entry) => {
        const role = entry.message?.role === "assistant" ? "Assistant" : "User";
        const text = extractContentText(entry.message?.content);
        return text ? `${role}: ${text}` : "";
      })
      .filter(Boolean);

    if (messages.length === 0) {
      return "";
    }
    return messages.slice(-historyTurns).join("\n");
  } catch {
    return "";
  }
}

async function persistLatentMasTurn(params: {
  prompt: string;
  finalText: string;
  sessionId: string;
  sessionFile: string;
  sessionCwd: string;
  provider: string;
  model: string;
  usage?: {
    input?: number;
    output?: number;
    total?: number;
  };
}): Promise<void> {
  const hadSessionFile = await fs
    .access(params.sessionFile)
    .then(() => true)
    .catch(() => false);
  const sessionManager = SessionManager.open(params.sessionFile);
  await prepareSessionManagerForRun({
    sessionManager,
    sessionFile: params.sessionFile,
    hadSessionFile,
    sessionId: params.sessionId,
    cwd: params.sessionCwd,
  });

  sessionManager.appendMessage({
    role: "user",
    content: params.prompt,
    timestamp: Date.now(),
  });
  sessionManager.appendMessage({
    role: "assistant",
    content: [{ type: "text", text: params.finalText }],
    api: "openai-responses",
    provider: params.provider,
    model: params.model,
    usage: {
      input: params.usage?.input ?? 0,
      output: params.usage?.output ?? 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: params.usage?.total ?? (params.usage?.input ?? 0) + (params.usage?.output ?? 0),
      cost: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        total: 0,
      },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  });
  emitSessionTranscriptUpdate(params.sessionFile);
}

export async function runLatentMasAgent(params: {
  sessionId: string;
  sessionFile: string;
  workspaceDir: string;
  cfg: OpenClawConfig;
  agentDir: string;
  provider: string;
  model: string;
  prompt: string;
  extraSystemPrompt?: string;
  thinkLevel?: ThinkLevel;
  streamParams?: AgentStreamParams;
  timeoutMs?: number;
  latentConfig?: ResolvedLatentMasConfig;
}): Promise<EmbeddedPiRunResult> {
  const latentConfig =
    params.latentConfig ??
    resolveLatentMasConfig({
      provider: params.provider,
      modelId: params.model,
      cfg: params.cfg,
      agentDir: params.agentDir,
      thinkLevel: params.thinkLevel,
      streamParams: params.streamParams,
    });
  if (!latentConfig) {
    throw new Error(`Model ${params.provider}/${params.model} is not configured for LatentMAS.`);
  }

  const startedAt = Date.now();
  const sessionContext = await buildSessionContext(params.sessionFile, latentConfig.historyTurns);
  const bridge = await runLatentMasBridge({
    config: latentConfig,
    prompt: params.prompt,
    extraSystemPrompt: params.extraSystemPrompt,
    sessionContext,
    timeoutMs: params.timeoutMs,
    workspaceDir: params.workspaceDir,
  });
  const finalText = bridge.text.trim();
  await persistLatentMasTurn({
    prompt: params.prompt,
    finalText,
    sessionId: params.sessionId,
    sessionFile: params.sessionFile,
    sessionCwd: params.workspaceDir,
    provider: params.provider,
    model: params.model,
    usage: bridge.usage,
  });

  return {
    payloads: [{ text: finalText }],
    meta: {
      durationMs: Date.now() - startedAt,
      agentMeta: {
        sessionId: params.sessionId,
        provider: params.provider,
        model: params.model,
        promptTokens: bridge.usage?.input,
        usage: {
          input: bridge.usage?.input,
          output: bridge.usage?.output,
          total: bridge.usage?.total,
        },
      },
      stopReason: "completed",
    },
  };
}
