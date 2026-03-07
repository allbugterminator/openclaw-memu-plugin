import { Extension } from "@openclaw/sdk";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { execa } from "execa";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface MemUConfig {
  mode: "local" | "cloud";
  cloud?: {
    apiKey: string;
    endpoint?: string;
  };
  local?: {
    storage: "file" | "postgres";
    dataDir?: string;
    postgres?: {
      host: string;
      port: number;
      database: string;
      user: string;
      password: string;
    };
  };
  llm?: {
    provider: "openai" | "anthropic" | "volcengine";
    apiKey: string;
    model?: string;
    endpoint?: string;
  };
}

interface MemUResult {
  success: boolean;
  data?: any;
  error?: string;
}

interface MemoryRecord {
  id: string;
  content: string;
  modality: "conversation" | "document" | "image" | "video" | "audio";
  userId?: string;
  timestamp: number;
  metadata?: Record<string, any>;
  embedding?: number[];
}

class MemUClient {
  private config: MemUConfig;
  private pythonPath: string;
  private servicePath: string;

  constructor(config: Partial<MemUConfig> = {}) {
    this.config = this.loadConfig(config);
    this.pythonPath = process.env.OPENCLAW_PYTHON_PATH || "python3";
    this.servicePath = path.join(__dirname, "..", "memu", "app.py");
  }

  private loadConfig(override: Partial<MemUConfig>): MemUConfig {
    const defaultConfig: MemUConfig = {
      mode: "local",
      local: {
        storage: "file",
        dataDir: path.join(process.env.HOME || "/tmp", ".memu", "data"),
      },
    };

    // Load from environment variables
    const envConfig: Partial<MemUConfig> = {};
    
    if (process.env.OPENCLAW_MEMU_MODE) {
      envConfig.mode = process.env.OPENCLAW_MEMU_MODE as "local" | "cloud";
    }

    if (process.env.OPENCLAW_MEMU_CLOUD_API_KEY) {
      envConfig.cloud = {
        apiKey: process.env.OPENCLAW_MEMU_CLOUD_API_KEY,
        endpoint: process.env.OPENCLAW_MEMU_CLOUD_ENDPOINT,
      };
    }

    if (process.env.OPENCLAW_MEMU_LOCAL_STORAGE) {
      envConfig.local = {
        ...envConfig.local,
        storage: process.env.OPENCLAW_MEMU_LOCAL_STORAGE as "file" | "postgres",
      };
    }

    if (process.env.OPENCLAW_MEMU_LOCAL_DATA_DIR) {
      envConfig.local = {
        ...envConfig.local,
        dataDir: process.env.OPENCLAW_MEMU_LOCAL_DATA_DIR,
      };
    }

    if (process.env.OPENCLAW_MEMU_LLM_PROVIDER) {
      envConfig.llm = {
        provider: process.env.OPENCLAW_MEMU_LLM_PROVIDER as any,
        apiKey: process.env.OPENCLAW_MEMU_LLM_API_KEY || "",
        model: process.env.OPENCLAW_MEMU_LLM_MODEL,
        endpoint: process.env.OPENCLAW_MEMU_LLM_ENDPOINT,
      };
    }

    return { ...defaultConfig, ...envConfig, ...override };
  }

  private async runPython(command: string, args: string[] = []): Promise<MemUResult> {
    try {
      const { stdout, stderr } = await execa(this.pythonPath, [this.servicePath, command, ...args], {
        env: {
          ...process.env,
          OPENCLAW_MEMU_CONFIG: JSON.stringify(this.config),
        },
        timeout: 30000,
      });

      if (stderr) {
        console.warn("MemU stderr:", stderr);
      }

      try {
        return JSON.parse(stdout);
      } catch (e) {
        return {
          success: false,
          error: `Invalid JSON response: ${stdout}`,
        };
      }
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Unknown error",
      };
    }
  }

  async memorize(content: string, modality: string = "conversation", userId?: string): Promise<MemUResult> {
    return this.runPython("memorize", [modality, content, userId || ""]);
  }

  async retrieve(queries: { role: string; content: { text: string } }[], method: "rag" | "llm" = "rag", userId?: string): Promise<MemUResult> {
    const queriesJson = JSON.stringify(queries);
    return this.runPython("retrieve", [method, queriesJson, userId || ""]);
  }

  async search(query: string, userId?: string): Promise<MemUResult> {
    return this.runPython("search", [query, userId || ""]);
  }

  async cloudMemorize(content: string, modality: string = "conversation"): Promise<MemUResult> {
    return this.runPython("cloud-memorize", [content, modality]);
  }

  async cloudRetrieve(queries: { role: string; content: { text: string } }[], method: "rag" | "llm" = "rag"): Promise<MemUResult> {
    const queriesJson = JSON.stringify(queries);
    return this.runPython("cloud-retrieve", [method, queriesJson]);
  }
}

const extension = new Extension({
  name: "memu",
  description: "MemU proactive memory plugin for OpenClaw",
  version: "1.0.0",
});

let memuClient: MemUClient;

extension.on("init", async () => {
  memuClient = new MemUClient();
  extension.logger.info("MemU plugin initialized");
});

extension.registerTool({
  name: "memu_memorize",
  description: "Store information in memU proactive memory. Use this to remember facts, preferences, skills, and important context from conversations or documents.",
  parameters: {
    type: "object",
    required: ["content"],
    properties: {
      content: {
        type: "string",
        description: "The content to memorize (conversation text, document content, or fact to remember)",
      },
      modality: {
        type: "string",
        enum: ["conversation", "document", "image", "video", "audio"],
        default: "conversation",
        description: "Type of content being memorized",
      },
      user_id: {
        type: "string",
        description: "Optional user identifier to scope the memory",
      },
    },
  },
  handler: async (args) => {
    const { content, modality = "conversation", user_id } = args;
    return memuClient.memorize(content, modality, user_id);
  },
});

extension.registerTool({
  name: "memu_retrieve",
  description: "Retrieve relevant memories from memU. Use this to fetch context, facts, preferences, and learned skills before responding to user queries.",
  parameters: {
    type: "object",
    required: ["query_text"],
    properties: {
      query_text: {
        type: "string",
        description: "The search query text to find relevant memories",
      },
      method: {
        type: "string",
        enum: ["rag", "llm"],
        default: "rag",
        description: "Retrieval method: 'rag' for fast embedding-based, 'llm' for deep reasoning",
      },
      user_id: {
        type: "string",
        description: "Optional user identifier to scope the search",
      },
    },
  },
  handler: async (args) => {
    const { query_text, method = "rag", user_id } = args;
    const queries = [{ role: "user", content: { text: query_text } }];
    return memuClient.retrieve(queries, method, user_id);
  },
});

extension.registerTool({
  name: "memu_search",
  description: "Quick search for specific facts or preferences in memU memory.",
  parameters: {
    type: "object",
    required: ["query"],
    properties: {
      query: {
        type: "string",
        description: "Search query text",
      },
      user_id: {
        type: "string",
        description: "Optional user identifier to scope the search",
      },
    },
  },
  handler: async (args) => {
    const { query, user_id } = args;
    return memuClient.search(query, user_id);
  },
});

export default extension;
