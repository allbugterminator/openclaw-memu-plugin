import { randomUUID } from "crypto";
import pg from "pg";
const { Pool } = pg;
import fetch from "node-fetch";

// 配置
const config = {
  storageType: "postgres",
  postgresConnectionString: "postgresql://postgres:postgres@localhost:5432/memu",
  autoLearn: true,
  proactiveRetrieval: true,
  isolationMode: "agent",
  llmApiKey: "",
  llmBaseUrl: "",
  embeddingModel: "text-embedding-3-small",
  embeddingApiKey: "",
  embeddingBaseUrl: ""
};

// PostgreSQL连接池
let pool: pg.Pool | null = null;

// 初始化PostgreSQL
async function initPostgres() {
  if (pool) return;
  
  pool = new Pool({
    connectionString: config.postgresConnectionString,
  });

  // 创建表（如果不存在）
  await pool.query(`
    CREATE EXTENSION IF NOT EXISTS vector;
    
    CREATE TABLE IF NOT EXISTS memories (
      id UUID PRIMARY KEY,
      content TEXT NOT NULL,
      metadata JSONB DEFAULT '{}',
      embedding vector(1536),
      timestamp BIGINT NOT NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
    CREATE INDEX IF NOT EXISTS idx_memories_metadata ON memories USING GIN(metadata);
  `);
  
  console.log("✅ memU with PostgreSQL initialized successfully");
}

// 初始化服务
async function initServices() {
  if (config.storageType === "postgres") {
    await initPostgres();
  }
}

// 调用外部向量模型API生成嵌入
async function generateEmbedding(text: string): Promise<number[]> {
  // 使用配置的向量模型参数，优先使用embedding专用配置，否则fallback到llm配置
  const apiKey = config.embeddingApiKey || config.llmApiKey || "your-api-key";
  const baseUrl = config.embeddingBaseUrl || config.llmBaseUrl || "http://localhost:8000/v1";
  const model = config.embeddingModel || "text-embedding-3-small";

  try {
    const response = await fetch(`${baseUrl}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: model,
        input: text.replace(/\n/g, " "),
        encoding_format: "float"
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error("Embedding API error:", response.status, error);
      throw new Error(`Embedding API request failed: ${response.status}`);
    }

    const data = await response.json() as any;
    return data.data[0].embedding;
  } catch (error) {
    console.error("Failed to generate embedding, falling back to local method:", error);
    // 降级到本地伪嵌入生成
    const hash = text.split('').reduce((acc, char) => {
      return char.charCodeAt(0) + ((acc << 5) - acc);
    }, 0);
    
    const embedding: number[] = [];
    for (let i = 0; i < 1536; i++) {
      const seed = hash * (i + 1);
      const value = Math.abs(Math.sin(seed)) * 2 - 1;
      embedding.push(parseFloat(value.toFixed(8)));
    }
    
    return embedding;
  }
}

// 内部存储记忆函数
async function storeMemory(text: string, metadata: any = {}): Promise<any> {
  try {
    if (!text.trim()) {
      return { success: false, error: "text cannot be empty" };
    }

    await initServices();
    const id = randomUUID();
    const timestamp = Date.now();
    const embedding = await generateEmbedding(text);
    
    // 向量格式转换：直接转为PostgreSQL vector支持的格式
    const embeddingStr = `[${embedding.join(',')}]`;
    
    await pool!.query(
      `INSERT INTO memories (id, content, metadata, embedding, timestamp)
       VALUES ($1, $2, $3, $4::vector, $5)`,
      [id, text, JSON.stringify(metadata), embeddingStr, timestamp]
    );

    return { 
      success: true, 
      data: { memoryId: id, message: "Memory stored successfully", storedText: text } 
    };
  } catch (error) {
    console.error("storeMemory error:", error);
    return { success: false, error: (error as Error).message };
  }
}

// 内部检索记忆函数
async function retrieveMemories(queryText: string, limit: number = 5, filter: any = {}): Promise<any> {
  try {
    if (!queryText.trim()) {
      return { success: false, error: "query_text cannot be empty" };
    }

    await initServices();
    const queryEmbedding = await generateEmbedding(queryText);
    const embeddingStr = `[${queryEmbedding.join(',')}]`;
    
    // 构建过滤条件
    let filterClause = "";
    const filterParams: any[] = [];
    
    if (filter && Object.keys(filter).length > 0) {
      const conditions = Object.entries(filter).map(([key, value], index) => {
        filterParams.push(value);
        return `metadata->>'${key}' = $${index + 3}`;
      });
      filterClause = "AND " + conditions.join(" AND ");
    }
    
    const result = await pool!.query(
      `SELECT id, content, metadata, timestamp,
              1 - (embedding <=> $1::vector) as similarity
       FROM memories
       WHERE 1=1 ${filterClause}
       ORDER BY embedding <=> $1::vector
       LIMIT $2`,
      [embeddingStr, limit, ...filterParams]
    );

    return { 
      success: true, 
      data: { 
        memories: result.rows,
        count: result.rows.length
      } 
    };
  } catch (error) {
    console.error("retrieveMemories error:", error);
    return { success: false, error: (error as Error).message };
  }
}

// 插件注册函数
function register(api: any) {
  // 加载配置
  if (api.pluginConfig) {
    Object.assign(config, api.pluginConfig);
  }

  // 覆盖memu_memorize工具
  api.registerTool({
    name: "memu_memorize",
    description: "Store a memory (PostgreSQL implementation)",
    parameters: {
      type: "object",
      required: ["text"],
      properties: {
        text: { type: "string" },
        metadata: { type: "object", default: {} }
      }
    },
    execute: async (a: any, b: any, c: any): Promise<any> => {
      console.log("memu_memorize arguments:", {a, b, c});
      // 尝试所有可能的参数位置
      let text = "";
      let metadata = {};
      
      if (a && typeof a === "object") {
        text = a.text || a.content || "";
        metadata = a.metadata || a.meta || {};
      }
      if (!text && b && typeof b === "object") {
        text = b.text || b.content || "";
        metadata = b.metadata || b.meta || {};
      }
      if (!text && typeof a === "string") text = a;
      if (!text && typeof b === "string") text = b;
      if (!text && typeof c === "string") text = c;
      
      return storeMemory(text, metadata);
    }
  });

  // 覆盖memu_retrieve工具
  api.registerTool({
    name: "memu_retrieve",
    description: "Retrieve memories (PostgreSQL implementation)",
    parameters: {
      type: "object",
      required: ["query_text"],
      properties: {
        query_text: { type: "string" },
        limit: { type: "number", default: 5 },
        filter: { type: "object", default: {} }
      }
    },
    execute: async (a: any, b: any, c: any): Promise<any> => {
      console.log("memu_retrieve arguments:", {a, b, c});
      // 尝试所有可能的参数位置
      let queryText = "";
      let limit = 5;
      let filter = {};
      
      if (a && typeof a === "object") {
        queryText = a.query_text || a.query || "";
        limit = a.limit || a.top_k || 5;
        filter = a.filter || {};
      }
      if (!queryText && b && typeof b === "object") {
        queryText = b.query_text || b.query || "";
        limit = b.limit || b.top_k || 5;
        filter = b.filter || {};
      }
      if (!queryText && typeof a === "string") queryText = a;
      if (!queryText && typeof b === "string") queryText = b;
      
      return retrieveMemories(queryText, limit, filter);
    }
  });

  // 覆盖memu_search工具
  api.registerTool({
    name: "memu_search",
    description: "Search memories (PostgreSQL implementation)",
    parameters: {
      type: "object",
      required: ["query"],
      properties: {
        query: { type: "string" }
      }
    },
    execute: async (params: any): Promise<any> => {
      const query = params.query || params.query_text || "";
      return retrieveMemories(query, 5, {});
    }
  });

  // 自动学习 - 使用agent_end事件
  if (config.autoLearn) {
    api.on("agent_end", async (event, ctx) => {
      console.log("[memu] agent_end hook triggered", { success: event.success, messageCount: event.messages?.length });
      if (!event.success || !event.messages) return;
      
      try {
        let lastUserQuery = "";
        const agentId = ctx.agentId || "default";
        const isolationMode = config.isolationMode || "none";
        
        for (const msg of event.messages) {
          if (!msg || !msg.content) continue;
          
          const content = typeof msg.content === "string" ? msg.content : 
            msg.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n");
          
          if (msg.role === "user") {
            lastUserQuery = content;
          } else if (msg.role === "assistant" && lastUserQuery) {
            const metadata: any = {
              type: "conversation",
              timestamp: Date.now()
            };
            
            // 根据隔离模式添加对应的隔离字段
            if (isolationMode.includes("agent") || isolationMode === "agent") {
              metadata.agentId = agentId;
            }
            if (isolationMode.includes("user") || isolationMode === "user") {
              metadata.userId = ctx.requesterSenderId;
            }
            if (isolationMode.includes("session") || isolationMode === "session") {
              metadata.sessionId = ctx.sessionId;
            }
            
            console.log("[memu] Auto-learning conversation:", lastUserQuery.substring(0, 50) + "...");
            await storeMemory(`用户: ${lastUserQuery}\n助手: ${content}`, metadata);
            
            lastUserQuery = "";
          }
        }
      } catch (error) {
        console.error("[memu] Auto-learn error:", error);
      }
    });
  }

  // 主动检索钩子 - 使用before_agent_start事件
  if (config.proactiveRetrieval) {
    api.on("before_agent_start", async (event, ctx) => {
      console.log("[memu] before_agent_start hook triggered");
      try {
        const query = event.messages?.filter((m: any) => m.role === "user")
          .map((m: any) => typeof m.content === "string" ? m.content : 
            m.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n"))
          .join("\n");
        
        if (!query) {
          console.log("[memu] No user query found for proactive retrieval");
          return;
        }
        
        const agentId = ctx.agentId || "default";
        const isolationMode = config.isolationMode || "none";
        
        const filter: any = {};
        if (isolationMode.includes("agent") || isolationMode === "agent") {
          filter.agentId = agentId;
        }
        if (isolationMode.includes("user") || isolationMode === "user") {
          filter.userId = ctx.requesterSenderId;
        }
        if (isolationMode.includes("session") || isolationMode === "session") {
          filter.sessionId = ctx.sessionId;
        }
        
        console.log("[memu] Proactive retrieval for query:", query.substring(0, 50) + "...");
        const result = await retrieveMemories(query, 3, filter);
        
        if (result.success && result.data.memories.length > 0) {
          // 将记忆添加到prependContext中
          const memoriesText = result.data.memories.map((m: any) => m.content).join("\n\n");
          event.prependContext = event.prependContext ? 
            `${event.prependContext}\n\n相关记忆:\n${memoriesText}` : 
            `相关记忆:\n${memoriesText}`;
          console.log("✅ [memu] Proactively loaded", result.data.memories.length, "memories");
        } else {
          console.log("[memu] No relevant memories found");
        }
      } catch (error) {
        console.error("[memu] Proactive retrieval error:", error);
      }
    });
  }

  // 激活
  initServices().catch(console.error);
}

export default register;
