import { spawn } from "child_process";
import { writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { randomUUID } from "crypto";

interface MemUConfig {
  provider: "cloud" | "self-hosted";
  cloudApiKey?: string;
  storageType?: "inmemory" | "postgres";
  postgresConnectionString?: string;
  llmProvider?: "openai" | "openrouter" | "custom";
  llmApiKey?: string;
  llmBaseUrl?: string;
  llmModel?: string;
  embeddingModel?: string;
  autoLearn?: boolean;
  proactiveRetrieval?: boolean;
}

interface MemUResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

interface ToolParams {
  [key: string]: unknown;
}

let memuClient: MemUClient | null = null;
let pluginApi: {
  registerTool: (tool: unknown) => void;
  registerGatewayMethod: (name: string, handler: (params: unknown) => Promise<unknown>) => void;
  logger: { info: (msg: string) => void; error: (msg: string) => void };
  config: Record<string, unknown>;
} | null = null;

function getConfig(): MemUConfig {
  if (!pluginApi) return { provider: "cloud" };
  return (pluginApi.config.plugins?.entries as Record<string, { config?: MemUConfig }>)?.memu?.config as MemUConfig || { provider: "cloud" };
}

class MemUClient {
  private config: MemUConfig;
  private pythonPath: string;
  private workDir: string;

  constructor(config: MemUConfig) {
    this.config = config;
    this.workDir = join(homedir(), ".openclaw", "memu-data");
    
    if (!existsSync(this.workDir)) {
      mkdirSync(this.workDir, { recursive: true });
    }
    
    this.pythonPath = process.platform === "win32" ? "python" : "python3";
  }

  private async runPython(script: string, args: string[] = [], content?: string): Promise<MemUResult> {
    return new Promise((resolve) => {
      const scriptPath = join(this.workDir, "memu_wrapper.py");
      
      const wrapperScript = this.generateWrapperScript(script);
      writeFileSync(scriptPath, wrapperScript, "utf-8");

      const proc = spawn(this.pythonPath, [scriptPath, ...args], {
        env: {
          ...process.env,
          OPENCLAW_MEMU_CONFIG: JSON.stringify(this.config),
          OPENCLAW_MEMU_CONTENT: content || "",
        },
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      proc.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      proc.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      proc.on("close", (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve({ success: true, data: result });
          } catch {
            resolve({ success: true, data: stdout });
          }
        } else {
          resolve({ success: false, error: stderr || `Process exited with code ${code}` });
        }
      });

      proc.on("error", (err) => {
        resolve({ success: false, error: err.message });
      });
    });
  }

  private generateWrapperScript(userScript: string): string {
    return `
import sys
import json
import os
import tempfile
import uuid

config = json.loads(os.environ.get("OPENCLAW_MEMU_CONFIG", "{}"))

def get_llm_profile():
    provider = config.get("llmProvider", "openai")
    if provider == "openai":
        return {
            "default": {
                "base_url": "https://api.openai.com/v1",
                "api_key": config.get("llmApiKey", os.environ.get("OPENAI_API_KEY", "")),
                "chat_model": config.get("llmModel", "gpt-4o"),
            }
        }
    elif provider == "openrouter":
        return {
            "default": {
                "provider": "openrouter",
                "client_backend": "httpx",
                "base_url": "https://openrouter.ai",
                "api_key": config.get("llmApiKey", os.environ.get("OPENROUTER_API_KEY", "")),
                "chat_model": config.get("llmModel", "anthropic/claude-3.5-sonnet"),
            }
        }
    elif provider == "custom":
        return {
            "default": {
                "base_url": config.get("llmBaseUrl", "http://localhost:8000/v1"),
                "api_key": config.get("llmApiKey", ""),
                "chat_model": config.get("llmModel", "gpt-4o"),
            }
        }
    return {}

def get_database_config():
    storage = config.get("storageType", "inmemory")
    if storage == "postgres":
        return {
            "metadata_store": {"provider": "postgres", "connection_string": config.get("postgresConnectionString", "")},
            "vector_store": {"provider": "postgres", "connection_string": config.get("postgresConnectionString", "")}
        }
    return {"metadata_store": {"provider": "inmemory"}}

async def memorize(content="", modality="conversation", user_id=None):
    try:
        from memu.app import MemoryService
        
        llm_profiles = get_llm_profile()
        db_config = get_database_config()
        
        service = MemoryService(
            llm_profiles=llm_profiles,
            database_config=db_config
        )
        
        # Create temp file for content since memu expects a file path
        temp_file = os.path.join(tempfile.gettempdir(), f"memu_{uuid.uuid4()}.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        user = {"user_id": user_id} if user_id else {}
        result = await service.memorize(
            resource_url=temp_file,
            modality=modality,
            user=user
        )
        
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass
        
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def retrieve(queries, method="rag", user_id=None):
    try:
        from memu.app import MemoryService
        
        llm_profiles = get_llm_profile()
        db_config = get_database_config()
        
        service = MemoryService(
            llm_profiles=llm_profiles,
            database_config=db_config,
            retrieve_config={"method": method}
        )
        
        # user_id goes in 'where' clause
        where = {"user_id": user_id} if user_id else {}
        result = await service.retrieve(
            queries=queries,
            where=where
        )
        
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def cloud_memorize(content, modality="conversation"):
    try:
        import requests
        
        api_key = config.get("cloudApiKey", os.environ.get("MEMU_API_KEY", ""))
        if not api_key:
            return {"success": False, "error": "No API key provided"}
        
        response = requests.post(
            "https://api.memu.so/api/v3/memory/memorize",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"content": content, "modality": modality}
        )
        
        if response.ok:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def cloud_retrieve(queries, method="rag"):
    try:
        import requests
        
        api_key = config.get("cloudApiKey", os.environ.get("MEMU_API_KEY", ""))
        if not api_key:
            return {"success": False, "error": "No API key provided"}
        
        response = requests.post(
            "https://api.memu.so/api/v3/memory/retrieve",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"queries": queries, "method": method}
        )
        
        if response.ok:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

${userScript}

if __name__ == "__main__":
    import asyncio
    import sys
    
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    args = sys.argv[2:]
    
    result = {}
    
    if command == "memorize":
        content = args[0] if args else ""
        modality = args[1] if len(args) > 1 else "conversation"
        user_id = args[2] if len(args) > 2 and args[2] else None
        result = asyncio.run(memorize(content, modality, user_id))
    elif command == "retrieve":
        queries_json = args[0] if args else "[]"
        method = args[1] if len(args) > 1 else "rag"
        user_id = args[2] if len(args) > 2 else None
        queries = json.loads(queries_json)
        result = asyncio.run(retrieve(queries, method, user_id))
    elif command == "cloud-memorize":
        content = args[0] if args else ""
        modality = args[1] if len(args) > 1 else "conversation"
        result = asyncio.run(cloud_memorize(content, modality))
    elif command == "cloud-retrieve":
        queries_json = args[0] if args else "[]"
        method = args[1] if len(args) > 1 else "rag"
        queries = json.loads(queries_json)
        result = asyncio.run(cloud_retrieve(queries, method))
    else:
        result = {"success": False, "error": f"Unknown command: {command}"}
    
    print(json.dumps(result))
`;
  }

  async memorize(content: string, modality: string = "conversation", userId?: string): Promise<MemUResult> {
    return this.runPython("memorize", [content, modality, userId || ""]);
  }

  async retrieve(queries: { role: string; content: { text: string } }[], method: "rag" | "llm" = "rag", userId?: string): Promise<MemUResult> {
    const queriesJson = JSON.stringify(queries);
    return this.runPython("retrieve", [queriesJson, method, userId || ""]);
  }

  async cloudMemorize(content: string, modality: string = "conversation"): Promise<MemUResult> {
    return this.runPython("cloud-memorize", [content, modality]);
  }

  async cloudRetrieve(queries: { role: string; content: { text: string } }[], method: "rag" | "llm" = "rag"): Promise<MemUResult> {
    const queriesJson = JSON.stringify(queries);
    return this.runPython("cloud-retrieve", [queriesJson, method]);
  }
}

function getClient(): MemUClient {
  if (!memuClient) {
    const config = getConfig();
    memuClient = new MemUClient(config);
  }
  return memuClient;
}

export default function (api: {
  registerTool: (tool: unknown) => void;
  registerGatewayMethod: (name: string, handler: (params: unknown) => Promise<unknown>) => void;
  logger: { info: (msg: string) => void; error: (msg: string) => void };
  config: Record<string, unknown>;
}) {
  pluginApi = api;

  api.registerTool({
    name: "memu_memorize",
    description: "Store information in memU proactive memory. Use this to remember facts, preferences, skills, and important context from conversations or documents.",
    parameters: {
      type: "object",
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
      required: ["content"],
    },
    async execute(_id: string, params: ToolParams) {
      try {
        const client = getClient();
        const { content, modality = "conversation", user_id } = params as {
          content: string;
          modality?: string;
          user_id?: string;
        };

        if (client["config"].provider === "cloud") {
          const result = await client.cloudMemorize(content, modality);
          if (!result.success) {
            return { content: [{ type: "text", text: `Error: ${result.error}` }] };
          }
          return { 
            content: [{ type: "text", text: `Content memorized to memU Cloud. Result: ${JSON.stringify(result.data)}` }]
          };
        } else {
          const result = await client.memorize(content, modality, user_id);
          if (!result.success) {
            return { content: [{ type: "text", text: `Error: ${result.error}` }] };
          }
          return { 
            content: [{ type: "text", text: `Content memorized to local memU. Result: ${JSON.stringify(result.data)}` }]
          };
        }
      } catch (error) {
        api.logger.error(`memu_memorize error: ${error}`);
        return { content: [{ type: "text", text: `Error: ${String(error)}` }] };
      }
    },
  });

  api.registerTool({
    name: "memu_retrieve",
    description: "Retrieve relevant memories from memU. Use this to fetch context, facts, preferences, and learned skills before responding to user queries.",
    parameters: {
      type: "object",
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
      required: ["query_text"],
    },
    async execute(_id: string, params: ToolParams) {
      try {
        const client = getClient();
        const { query_text, method = "rag", user_id } = params as {
          query_text: string;
          method?: "rag" | "llm";
          user_id?: string;
        };

        const queries = [{ role: "user", content: { text: query_text } }];

        if (client["config"].provider === "cloud") {
          const result = await client.cloudRetrieve(queries, method);
          if (!result.success) {
            return { content: [{ type: "text", text: `Error: ${result.error}` }] };
          }
          return { 
            content: [{ type: "text", text: `Retrieved memories: ${JSON.stringify(result.data)}` }]
          };
        } else {
          const result = await client.retrieve(queries, method, user_id);
          if (!result.success) {
            return { content: [{ type: "text", text: `Error: ${result.error}` }] };
          }
          return { 
            content: [{ type: "text", text: `Retrieved memories: ${JSON.stringify(result.data)}` }]
          };
        }
      } catch (error) {
        api.logger.error(`memu_retrieve error: ${error}`);
        return { content: [{ type: "text", text: `Error: ${String(error)}` }] };
      }
    },
  });

  api.registerTool({
    name: "memu_search",
    description: "Quick search for specific facts or preferences in memU memory.",
    parameters: {
      type: "object",
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
      required: ["query"],
    },
    async execute(_id: string, params: ToolParams) {
      try {
        const client = getClient();
        const { query, user_id } = params as {
          query: string;
          user_id?: string;
        };
        const queries = [{ role: "user", content: { text: query } }];

        const result = await client.retrieve(queries, "rag", user_id);
        if (!result.success) {
          return { content: [{ type: "text", text: `Error: ${result.error}` }] };
        }
        return { 
          content: [{ type: "text", text: `Search results: ${JSON.stringify(result.data)}` }]
        };
      } catch (error) {
        api.logger.error(`memu_search error: ${error}`);
        return { content: [{ type: "text", text: `Error: ${String(error)}` }] };
      }
    },
  });

  api.registerGatewayMethod("memu.status", async () => {
    return {
      ok: true,
      plugin: "memu",
      version: "1.0.0",
      description: "24/7 Proactive Memory for AI Agents",
    };
  });

  api.registerGatewayMethod("memu.health", async () => {
    try {
      const config = getConfig();
      return {
        ok: true,
        provider: config.provider,
        storage: config.storageType || "cloud",
      };
    } catch (error) {
      return {
        ok: false,
        error: String(error),
      };
    }
  });

  api.logger.info("memU plugin loaded - 24/7 Proactive Memory for AI Agents");
}

export const id = "memu";
export const name = "memU Memory Integration";
export const version = "1.0.0";
