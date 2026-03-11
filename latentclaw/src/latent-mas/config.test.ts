import { describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../config/config.js";
import { resolveLatentMasConfig } from "./config.js";

describe("resolveLatentMasConfig", () => {
  it("enables latent provider by default and maps think level to latent steps", () => {
    const config = resolveLatentMasConfig({
      provider: "latentmas",
      modelId: "Qwen/Qwen3-4B",
      cfg: {
        models: {
          providers: {
            latentmas: {
              baseUrl: "http://127.0.0.1/latent",
              api: "openai-responses",
              models: [{ id: "Qwen/Qwen3-4B", name: "Qwen", reasoning: true, input: ["text"] }],
            },
          },
        },
      } as OpenClawConfig,
      agentDir: "/tmp/agent",
      thinkLevel: "high",
    });

    expect(config).toMatchObject({
      hfModelName: "Qwen/Qwen3-4B",
      latentSteps: 16,
      prompt: "sequential",
      maxNewTokens: 1024,
      temperature: 0.6,
      topP: 0.95,
      latentSpaceRealign: true,
    });
  });

  it("uses compat overrides when latent mas is enabled on a non-latent provider", () => {
    const config = resolveLatentMasConfig({
      provider: "custom",
      modelId: "my-model",
      cfg: {
        models: {
          providers: {
            custom: {
              baseUrl: "http://127.0.0.1/custom",
              api: "openai-responses",
              models: [
                {
                  id: "my-model",
                  name: "My Model",
                  reasoning: true,
                  input: ["text"],
                  compat: {
                    latentMas: {
                      enabled: true,
                      modelName: "Qwen/Qwen3-14B",
                      latentSteps: 20,
                      prompt: "hierarchical",
                      maxNewTokens: 2048,
                      temperature: 0.3,
                      topP: 0.8,
                      device: "cuda:1",
                      thinkToken: true,
                      latentSpaceRealign: false,
                      torchDtype: "float16",
                      historyTurns: 4,
                    },
                  },
                },
              ],
            },
          },
        },
      } as OpenClawConfig,
      agentDir: "/tmp/agent",
      thinkLevel: "low",
      streamParams: { maxTokens: 256, temperature: 0.2 },
    });

    expect(config).toMatchObject({
      hfModelName: "Qwen/Qwen3-14B",
      latentSteps: 20,
      prompt: "hierarchical",
      maxNewTokens: 256,
      temperature: 0.2,
      topP: 0.8,
      device: "cuda:1",
      thinkToken: true,
      latentSpaceRealign: false,
      torchDtype: "float16",
      historyTurns: 4,
    });
  });

  it("returns null when the selected model is not latent-enabled", () => {
    const config = resolveLatentMasConfig({
      provider: "openai",
      modelId: "gpt-5.2",
      cfg: {
        models: {
          providers: {
            openai: {
              baseUrl: "https://api.openai.com/v1",
              api: "openai-responses",
              models: [{ id: "gpt-5.2", name: "GPT", reasoning: true, input: ["text"] }],
            },
          },
        },
      } as OpenClawConfig,
      agentDir: "/tmp/agent",
    });

    expect(config).toBeNull();
  });
});
