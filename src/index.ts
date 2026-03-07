import { Extension } from "@openclaw/sdk";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { execa } from "execa";
import { spawn } from "child_process";
import { writeFileSync, mkdirSync, existsSync, readFileSync } from "fs";
import { homedir } from "os";
import { randomUUID } from "crypto";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface MemUConfig {
  provider: "cloud" | "self-hosted";
  cloudApiKey?: string;
  cloudEndpoint?: string;
  storageType?: "inmemory" | "postgres" | "file";
  postgresConnectionString?: string;
  dataDir?: string;
  llmProvider?: "openai" | "openrouter" | "custom" | "anthropic" | "volcengine";
  llmApiKey?: string;
  llmBaseUrl?: string;
  llmModel?: string;
  embeddingProvider?: "openai" | "custom";
  embeddingApiKey?: string;
  embeddingBaseUrl?: string;
  embeddingModel?: string;
  autoLearn?: boolean;
  proactiveRetrieval?: boolean;
  retrievalTopK?: number;
  retrievalThreshold?: number;
  sessionTTL?: number;
  maxMemorySize?: number;
}

interface MemUResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

interface MemoryRecord {
  id: string;
  content: string;
  modality: "conversation" | "document" | "image" | "video" | "audio";
  userId?: string;
  sessionId?: string;
  timestamp: number;
  metadata?: Record<string, any>;
  embedding?: number[];
  score?: number;
}

interface RetrievalOptions {
  topK?: number;
  threshold?: number;
  includeEmbedding?: boolean;
  filter?: Record<string, any>;
}

class MemUClient {
  private config: MemUConfig;
  private pythonPath: string;
  private workDir: string;
  private servicePath: string;
  private wrapperPath: string;

  constructor(config: Partial<MemUConfig> = {}) {
    this.config = this.loadConfig(config);
    this.pythonPath = process.platform === "win32" ? "python" : "python3";
    this.workDir = this.config.dataDir || path.join(homedir(), ".openclaw", "memu-data");
    this.servicePath = path.join(__dirname, "..", "memu", "app.py");
    this.wrapperPath = path.join(this.workDir, "memu_wrapper.py");
    
    this.initWorkDir();
  }

  private loadConfig(override: Partial<MemUConfig>): MemUConfig {
    const defaultConfig: MemUConfig = {
      provider: "self-hosted",
      storageType: "file",
      autoLearn: false,
      proactiveRetrieval: false,
      retrievalTopK: 5,
      retrievalThreshold: 0.7,
      sessionTTL: 86400 * 7, // 7 days
      maxMemorySize: 10000,
    };

    // Load from environment variables
    const envConfig: Partial<MemUConfig> = {};
    
    if (process.env.OPENCLAW_MEMU_PROVIDER) {
      envConfig.provider = process.env.OPENCLAW_MEMU_PROVIDER as "cloud" | "self-hosted";
    }

    if (process.env.OPENCLAW_MEMU_CLOUD_API_KEY) {
      envConfig.cloudApiKey = process.env.OPENCLAW_MEMU_CLOUD_API_KEY;
    }

    if (process.env.OPENCLAW_MEMU_CLOUD_ENDPOINT) {
      envConfig.cloudEndpoint = process.env.OPENCLAW_MEMU_CLOUD_ENDPOINT;
    }

    if (process.env.OPENCLAW_MEMU_STORAGE_TYPE) {
      envConfig.storageType = process.env.OPENCLAW_MEMU_STORAGE_TYPE as "inmemory" | "postgres" | "file";
    }

    if (process.env.OPENCLAW_MEMU_POSTGRES_DSN) {
      envConfig.postgresConnectionString = process.env.OPENCLAW_MEMU_POSTGRES_DSN;
    }

    if (process.env.OPENCLAW_MEMU_DATA_DIR) {
      envConfig.dataDir = process.env.OPENCLAW_MEMU_DATA_DIR;
    }

    if (process.env.OPENCLAW_MEMU_LLM_PROVIDER) {
      envConfig.llmProvider = process.env.OPENCLAW_MEMU_LLM_PROVIDER as any;
    }

    if (process.env.OPENCLAW_MEMU_LLM_API_KEY) {
      envConfig.llmApiKey = process.env.OPENCLAW_MEMU_LLM_API_KEY;
    }

    if (process.env.OPENCLAW_MEMU_LLM_BASE_URL) {
      envConfig.llmBaseUrl = process.env.OPENCLAW_MEMU_LLM_BASE_URL;
    }

    if (process.env.OPENCLAW_MEMU_LLM_MODEL) {
      envConfig.llmModel = process.env.OPENCLAW_MEMU_LLM_MODEL;
    }

    if (process.env.OPENCLAW_MEMU_EMBEDDING_PROVIDER) {
      envConfig.embeddingProvider = process.env.OPENCLAW_MEMU_EMBEDDING_PROVIDER as any;
    }

    if (process.env.OPENCLAW_MEMU_EMBEDDING_API_KEY) {
      envConfig.embeddingApiKey = process.env.OPENCLAW_MEMU_EMBEDDING_API_KEY;
    }

    if (process.env.OPENCLAW_MEMU_EMBEDDING_BASE_URL) {
      envConfig.embeddingBaseUrl = process.env.OPENCLAW_MEMU_EMBEDDING_BASE_URL;
    }

    if (process.env.OPENCLAW_MEMU_EMBEDDING_MODEL) {
      envConfig.embeddingModel = process.env.OPENCLAW_MEMU_EMBEDDING_MODEL;
    }

    if (process.env.OPENCLAW_MEMU_AUTO_LEARN) {
      envConfig.autoLearn = process.env.OPENCLAW_MEMU_AUTO_LEARN === "true";
    }

    if (process.env.OPENCLAW_MEMU_PROACTIVE_RETRIEVAL) {
      envConfig.proactiveRetrieval = process.env.OPENCLAW_MEMU_PROACTIVE_RETRIEVAL === "true";
    }

    if (process.env.OPENCLAW_MEMU_RETRIEVAL_TOP_K) {
      envConfig.retrievalTopK = parseInt(process.env.OPENCLAW_MEMU_RETRIEVAL_TOP_K, 10);
    }

    if (process.env.OPENCLAW_MEMU_RETRIEVAL_THRESHOLD) {
      envConfig.retrievalThreshold = parseFloat(process.env.OPENCLAW_MEMU_RETRIEVAL_THRESHOLD);
    }

    return { ...defaultConfig, ...envConfig, ...override };
  }

  private initWorkDir(): void {
    if (!existsSync(this.workDir)) {
      mkdirSync(this.workDir, { recursive: true });
    }
  }

  private generateWrapperScript(command: string): string {
    return `
import sys
import json
import os
sys.path.insert(0, "${path.dirname(this.servicePath)}")

try:
    from app import MemU
    config = json.loads(os.environ.get("OPENCLAW_MEMU_CONFIG", "{}"))
    memu = MemU(config)
    
    args = sys.argv[1:]
    
    if command == "memorize":
        content = args[1] if len(args) > 1 else ""
        modality = args[0] if len(args) > 0 else "conversation"
        user_id = args[2] if len(args) > 2 else None
        result = memu.memorize(content, modality, user_id)
    elif command == "retrieve":
        method = args[0] if len(args) > 0 else "rag"
        queries = json.loads(args[1]) if len(args) > 1 else []
        user_id = args[2] if len(args) > 2 else None
        options = json.loads(args[3]) if len(args) > 3 else {}
        result = memu.retrieve(queries, method, user_id, options)
    elif command == "search":
        query = args[0] if len(args) > 0 else ""
        user_id = args[1] if len(args) > 1 else None
        options = json.loads(args[2]) if len(args) > 2 else {}
        result = memu.search(query, user_id, options)
    elif command == "delete":
        memory_id = args[0] if len(args) > 0 else ""
        user_id = args[1] if len(args) > 1 else None
        result = memu.delete(memory_id, user_id)
    elif command == "list":
        user_id = args[0] if len(args) > 0 else None
        limit = int(args[1]) if len(args) > 1 else 100
        offset = int(args[2]) if len(args) > 2 else 0
        result = memu.list(user_id, limit, offset)
    elif command == "clear":
        user_id = args[0] if len(args) > 0 else None
        result = memu.clear(user_id)
    elif command == "stats":
        user_id = args[0] if len(args) > 0 else None
        result = memu.get_stats(user_id)
    else:
        result = {"success": False, "error": f"Unknown command: {command}"}
    
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
    sys.exit(1)
`.replace(/"/g, '\\"').replace(/\n/g, "\\n");
  }

  private async runPython(command: string, args: string[] = []): Promise<MemUResult> {
    try {
      if (this.config.provider === "cloud" && this.config.cloudApiKey) {
        // Cloud mode implementation
        const fetch = (await import("node-fetch")).default;
        const endpoint = this.config.cloudEndpoint || "https://api.memu.ai/v1";
        
        const response = await fetch(`${endpoint}/${command}`, {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${this.config.cloudApiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ args }),
          timeout: 30000,
        });

        if (!response.ok) {
          return {
            success: false,
            error: `Cloud API error: ${response.status} ${response.statusText}`,
          };
        }

        return await response.json() as MemUResult;
      }

      // Self-hosted mode
      const wrapperScript = `
import sys
import json
import os
sys.path.insert(0, "${path.dirname(this.servicePath)}")

try:
    from app import MemU
    config = json.loads(os.environ.get("OPENCLAW_MEMU_CONFIG", "{}"))
    memu = MemU(config)
    
    args = sys.argv[1:]
    
    ${command === "memorize" ? `
content = args[1] if len(args) > 1 else ""
modality = args[0] if len(args) > 0 else "conversation"
user_id = args[2] if len(args) > 2 else None
result = memu.memorize(content, modality, user_id)
` : command === "retrieve" ? `
method = args[0] if len(args) > 0 else "rag"
queries = json.loads(args[1]) if len(args) > 1 else []
user_id = args[2] if len(args) > 2 else None
options = json.loads(args[3]) if len(args) > 3 else {}
result = memu.retrieve(queries, method, user_id, options)
` : command === "search" ? `
query = args[0] if len(args) > 0 else ""
user_id = args[1] if len(args) > 1 else None
options = json.loads(args[2]) if len(args) > 2 else {}
result = memu.search(query, user_id, options)
` : command === "delete" ? `
memory_id = args[0] if len(args) > 0 else ""
user_id = args[1] if len(args) > 1 else None
result = memu.delete(memory_id, user_id)
` : command === "list" ? `
user_id = args[0] if len(args) > 0 else None
limit = int(args[1]) if len(args) > 1 else 100
offset = int(args[2]) if len(args) > 2 else 0
result = memu.list(user_id, limit, offset)
` : command === "clear" ? `
user_id = args[0] if len(args) > 0 else None
result = memu.clear(user_id)
` : command === "stats" ? `
user_id = args[0] if len(args) > 0 else None
result = memu.get_stats(user_id)
` : `
result = {"success": False, "error": f"Unknown command: {command}"}
`}
    
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
    sys.exit(1)
`;

      writeFileSync(this.wrapperPath, wrapperScript, "utf-8");

      const { stdout, stderr } = await execa(this.pythonPath, [this.wrapperPath, ...args], {
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

  // Core functionality
  async memorize(content: string, modality: string = "conversation", userId?: string, metadata?: Record<string, any>): Promise<MemUResult> {
    const args = [modality, content, userId || "", JSON.stringify(metadata || {})];
    return this.runPython("memorize", args);
  }

  async retrieve(queries: Array<{ role: string; content: { text: string } }>, method: "rag" | "llm" = "rag", userId?: string, options?: RetrievalOptions): Promise<MemUResult> {
    const args = [method, JSON.stringify(queries), userId || "", JSON.stringify(options || {})];
    return this.runPython("retrieve", args);
  }

  async search(query: string, userId?: string, options?: RetrievalOptions): Promise<MemUResult> {
    const args = [query, userId || "", JSON.stringify(options || {})];
    return this.runPython("search", args);
  }

  // Advanced functionality
  async delete(memoryId: string, userId?: string): Promise<MemUResult> {
    const args = [memoryId, userId || ""];
    return this.runPython("delete", args);
  }

  async list(userId?: string, limit: number = 100, offset: number = 0): Promise<MemUResult> {
    const args = [userId || "", limit.toString(), offset.toString()];
    return this.runPython("list", args);
  }

  async clear(userId?: string): Promise<MemUResult> {
    const args = [userId || ""];
    return this.runPython("clear", args);
  }

  async getStats(userId?: string): Promise<MemUResult> {
    const args = [userId || ""];
    return this.runPython("stats", args);
  }

  // Batch operations
  async batchMemorize(records: Array<Omit<MemoryRecord, "id" | "timestamp">>): Promise<MemUResult> {
    const results = [];
    for (const record of records) {
      const result = await this.memorize(record.content, record.modality, record.userId, record.metadata);
      results.push(result);
    }
    return {
      success: results.every(r => r.success),
      data: results,
    };
  }

  async batchDelete(memoryIds: string[], userId?: string): Promise<MemUResult> {
    const results = [];
    for (const id of memoryIds) {
      const result = await this.delete(id, userId);
      results.push(result);
    }
    return {
      success: results.every(r => r.success),
      data: results,
    };
  }

  // Auto-learn functionality
  async autoLearnFromConversation(messages: Array<{ role: string; content: string }>, userId?: string): Promise<MemUResult> {
    if (!this.config.autoLearn) {
      return {
        success: false,
        error: "Auto-learn is disabled in config",
      };
    }

    // Extract important information from conversation
    const importantPoints = await this.extractImportantPoints(messages);
    const results = [];

    for (const point of importantPoints) {
      const result = await this.memorize(point, "conversation", userId, {
        source: "auto-learn",
        conversationLength: messages.length,
      });
      results.push(result);
    }

    return {
      success: results.every(r => r.success),
      data: {
        learnedPoints: importantPoints.length,
        results,
      },
    };
  }

  private async extractImportantPoints(messages: Array<{ role: string; content: string }>): Promise<string[]> {
    // Use LLM to extract important points from conversation
    if (!this.config.llmApiKey) {
      return messages
        .filter(m => m.role === "user" || m.role === "assistant")
        .map(m => m.content)
        .filter(c => c.length > 10);
    }

    try {
      const fetch = (await import("node-fetch")).default;
      const prompt = `
Extract the most important facts, preferences, decisions, and information from the following conversation.
Return only a JSON array of strings, each being a single important point.
Do not include any other text or explanation.

Conversation:
${messages.map(m => `${m.role}: ${m.content}`).join("\n")}
`;

      const response = await fetch(this.config.llmBaseUrl || "https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.config.llmApiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: this.config.llmModel || "gpt-3.5-turbo",
          messages: [{ role: "user", content: prompt }],
          temperature: 0.1,
        }),
        timeout: 10000,
      });

      if (response.ok) {
        const data = await response.json() as any;
        const content = data.choices?.[0]?.message?.content || "[]";
        try {
          return JSON.parse(content);
        } catch {
          return content.split("\n").filter(line => line.trim().length > 0);
        }
      }
    } catch (e) {
      console.warn("Failed to extract important points with LLM:", e);
    }

    // Fallback: return all user messages
    return messages.filter(m => m.role === "user").map(m => m.content);
  }

  // Proactive retrieval
  async proactiveRetrieval(currentContext: string, userId?: string): Promise<MemUResult> {
    if (!this.config.proactiveRetrieval) {
      return {
        success: false,
        error: "Proactive retrieval is disabled in config",
      };
    }

    return this.search(currentContext, userId, {
      topK: this.config.retrievalTopK,
      threshold: this.config.retrievalThreshold,
    });
  }
}

const extension = new Extension({
  name: "memu",
  description: "MemU proactive memory plugin for OpenClaw with advanced features",
  version: "1.1.0",
});

let memuClient: MemUClient;

extension.on("init", async () => {
  memuClient = new MemUClient();
  extension.logger.info("MemU plugin initialized with advanced features");

  // Register gateway methods for internal use
  extension.registerGatewayMethod("memu.memorize", async (params: any) => {
    return memuClient.memorize(params.content, params.modality, params.userId, params.metadata);
  });

  extension.registerGatewayMethod("memu.retrieve", async (params: any) => {
    return memuClient.retrieve(params.queries, params.method, params.userId, params.options);
  });

  extension.registerGatewayMethod("memu.search", async (params: any) => {
    return memuClient.search(params.query, params.userId, params.options);
  });

  extension.registerGatewayMethod("memu.delete", async (params: any) => {
    return memuClient.delete(params.memoryId, params.userId);
  });

  extension.registerGatewayMethod("memu.list", async (params: any) => {
    return memuClient.list(params.userId, params.limit, params.offset);
  });

  extension.registerGatewayMethod("memu.clear", async (params: any) => {
    return memuClient.clear(params.userId);
  });

  extension.registerGatewayMethod("memu.stats", async (params: any) => {
    return memuClient.getStats(params.userId);
  });
});

// Core tools
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
      metadata: {
        type: "object",
        description: "Optional metadata to attach to the memory",
      },
    },
  },
  handler: async (args) => {
    const { content, modality = "conversation", user_id, metadata } = args;
    return memuClient.memorize(content, modality, user_id, metadata);
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
      top_k: {
        type: "number",
        description: "Number of results to return (default: 5)",
      },
      threshold: {
        type: "number",
        description: "Minimum similarity score threshold (0-1, default: 0.7)",
      },
    },
  },
  handler: async (args) => {
    const { query_text, method = "rag", user_id, top_k, threshold } = args;
    const queries = [{ role: "user", content: { text: query_text } }];
    const options: RetrievalOptions = {};
    if (top_k) options.topK = top_k;
    if (threshold) options.threshold = threshold;
    return memuClient.retrieve(queries, method, user_id, options);
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
      top_k: {
        type: "number",
        description: "Number of results to return (default: 5)",
      },
      threshold: {
        type: "number",
        description: "Minimum similarity score threshold (0-1, default: 0.7)",
      },
    },
  },
  handler: async (args) => {
    const { query, user_id, top_k, threshold } = args;
    const options: RetrievalOptions = {};
    if (top_k) options.topK = top_k;
    if (threshold) options.threshold = threshold;
    return memuClient.search(query, user_id, options);
  },
});

// Advanced tools
extension.registerTool({
  name: "memu_delete",
  description: "Delete a specific memory from memU by ID.",
  parameters: {
    type: "object",
    required: ["memory_id"],
    properties: {
      memory_id: {
        type: "string",
        description: "ID of the memory to delete",
      },
      user_id: {
        type: "string",
        description: "Optional user identifier to scope the deletion",
      },
    },
  },
  handler: async (args) => {
    const { memory_id, user_id } = args;
    return memuClient.delete(memory_id, user_id);
  },
});

extension.registerTool({
  name: "memu_list",
  description: "List all memories for a user with pagination.",
  parameters: {
    type: "object",
    properties: {
      user_id: {
        type: "string",
        description: "Optional user identifier to list memories for",
      },
      limit: {
        type: "number",
        description: "Number of results to return (default: 100)",
      },
      offset: {
        type: "number",
        description: "Offset for pagination (default: 0)",
      },
    },
  },
  handler: async (args) => {
    const { user_id, limit = 100, offset = 0 } = args;
    return memuClient.list(user_id, limit, offset);
  },
});

extension.registerTool({
  name: "memu_clear",
  description: "Clear all memories for a user. Use with caution!",
  parameters: {
    type: "object",
    properties: {
      user_id: {
        type: "string",
        description: "Optional user identifier to clear memories for",
      },
    },
  },
  handler: async (args) => {
    const { user_id } = args;
    return memuClient.clear(user_id);
  },
});

extension.registerTool({
  name: "memu_stats",
  description: "Get memory statistics for a user (total memories, storage usage, etc.)",
  parameters: {
    type: "object",
    properties: {
      user_id: {
        type: "string",
        description: "Optional user identifier to get stats for",
      },
    },
  },
  handler: async (args) => {
    const { user_id } = args;
    return memuClient.getStats(user_id);
  },
});

extension.registerTool({
  name: "memu_batch_memorize",
  description: "Batch store multiple memories at once.",
  parameters: {
    type: "object",
    required: ["records"],
    properties: {
      records: {
        type: "array",
        items: {
          type: "object",
          required: ["content"],
          properties: {
            content: { type: "string" },
            modality: { type: "string", enum: ["conversation", "document", "image", "video", "audio"], default: "conversation" },
            user_id: { type: "string" },
            metadata: { type: "object" },
          },
        },
        description: "Array of memory records to store",
      },
    },
  },
  handler: async (args) => {
    const { records } = args;
    return memuClient.batchMemorize(records);
  },
});

extension.registerTool({
  name: "memu_batch_delete",
  description: "Batch delete multiple memories by ID.",
  parameters: {
    type: "object",
    required: ["memory_ids"],
    properties: {
      memory_ids: {
        type: "array",
        items: { type: "string" },
        description: "Array of memory IDs to delete",
      },
      user_id: {
        type: "string",
        description: "Optional user identifier to scope the deletion",
      },
    },
  },
  handler: async (args) => {
    const { memory_ids, user_id } = args;
    return memuClient.batchDelete(memory_ids, user_id);
  },
});

export default extension;
