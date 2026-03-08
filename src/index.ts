import { randomUUID } from "crypto";
import pg from "pg";
const { Pool } = pg;

// 全局状态
let pool: pg.Pool | null = null;
let initialized = false;
let config: any = {};

// 简单的伪嵌入生成（1536维，模拟text-embedding-3-small的输出）
function generateEmbedding(text: string): number[] {
  const hash = text.split('').reduce((acc, char) => {
    return char.charCodeAt(0) + ((acc << 5) - acc);
  }, 0);
  
  // 生成1536维随机但稳定的向量
  const embedding: number[] = [];
  for (let i = 0; i < 1536; i++) {
    const seed = hash * (i + 1);
    const value = Math.abs(Math.sin(seed)) * 2 - 1; // 范围 [-1, 1]
    embedding.push(parseFloat(value.toFixed(8)));
  }
  
  return embedding;
}

// 初始化PostgreSQL连接和表结构
async function initPostgres() {
  if (pool) return pool;
  
  const connectionString = config.postgresConnectionString || "postgresql://postgres:postgres@localhost:5432/memu";
  pool = new Pool({ connectionString });
  
  // 创建表结构
  await pool.query(`
    CREATE TABLE IF NOT EXISTS memories (
      id TEXT PRIMARY KEY,
      content TEXT NOT NULL,
      metadata JSONB DEFAULT '{}'::JSONB,
      embedding vector(1536) NOT NULL,
      timestamp BIGINT NOT NULL
    );
  `);
  
  // 创建索引
  await pool.query(`
    CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories 
    USING hnsw (embedding vector_cosine_ops);
  `);
  
  console.log("✅ PostgreSQL initialized successfully");
  return pool;
}

// 初始化服务
async function initServices() {
  if (initialized) return;
  await initPostgres();
  initialized = true;
  console.log("✅ memU with PostgreSQL initialized successfully");
}

// 注册插件
function register(api: any) {
  config = api.config || {};

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
    execute: async (a: any, b: any, c: any) => {
      try {
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
        
        if (!text.trim()) {
          return { 
            success: false, 
            error: "text cannot be empty",
            debug: { args: [a, b, c], text, metadata }
          };
        }

        await initServices();
        const id = randomUUID();
        const timestamp = Date.now();
        const embedding = generateEmbedding(text);
        
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
        console.error("memu_memorize error:", error);
        return { 
          success: false, 
          error: (error as Error).message 
        };
      }
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
    execute: async (a: any, b: any, c: any) => {
      try {
        console.log("memu_retrieve arguments:", {a, b, c});
        // 尝试所有可能的参数位置
        let queryText = "";
        let limit = 5;
        let filter = {};
        
        if (a && typeof a === "object") {
          queryText = a.query_text || a.query || a.content || "";
          limit = a.limit || limit;
          filter = a.filter || {};
        }
        if (!queryText && b && typeof b === "object") {
          queryText = b.query_text || b.query || b.content || "";
          limit = b.limit || limit;
          filter = b.filter || {};
        }
        if (!queryText && typeof a === "string") queryText = a;
        if (!queryText && typeof b === "string") queryText = b;
        if (!queryText && typeof c === "string") queryText = c;
        
        if (!queryText.trim()) {
          return { 
            success: false, 
            error: "query_text cannot be empty",
            debug: { args: [a, b, c], queryText, limit, filter }
          };
        }

        await initServices();
        const queryEmbedding = generateEmbedding(queryText);
        const embeddingStr = `[${queryEmbedding.join(',')}]`;
        
        // 构建过滤条件
        let whereClauses: string[] = [];
        let queryParams: any[] = [embeddingStr, limit];
        let paramIndex = 3;
        
        for (const [key, value] of Object.entries(filter)) {
          whereClauses.push(`metadata->>'${key}' = $${paramIndex}`);
          queryParams.push(value);
          paramIndex++;
        }
        
        const whereClause = whereClauses.length > 0 ? `WHERE ${whereClauses.join(' AND ')}` : "";
        
        const result = await pool!.query(
          `SELECT id, content, metadata, timestamp,
                  1 - (embedding <=> $1::vector) as similarity
           FROM memories
           ${whereClause}
           ORDER BY embedding <=> $1::vector
           LIMIT $2`,
          queryParams
        );

        return { 
          success: true, 
          data: { 
            memories: result.rows,
            message: "Memories retrieved successfully",
            debug: { queryText, filter, count: result.rows.length }
          } 
        };
      } catch (error) {
        console.error("memu_retrieve error:", error);
        return { 
          success: false, 
          error: (error as Error).message 
        };
      }
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
    execute: async (params: any) => {
      return api.tools.get("memu_retrieve").execute(params);
    }
  });

  // 自动学习
  if (config.autoLearn) {
    api.on("agent_end", async (event: any) => {
      if (!event.success || !event.messages) return;
      
      try {
        let lastUserQuery = "";
        const agentId = event.agentId || "default";
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
              metadata.userId = event.userId;
            }
            if (isolationMode.includes("session") || isolationMode === "session") {
              metadata.sessionId = event.sessionId;
            }
            
            await api.tools.get("memu_memorize").execute({
              text: `用户: ${lastUserQuery}\n助手: ${content}`,
              metadata
            });
            
            lastUserQuery = "";
          }
        }
      } catch (error) {
        console.error("Auto-learn error:", error);
      }
    });
  }

  // 主动检索钩子
  if (config.proactiveRetrieval) {
    api.on("agent_start", async (event: any) => {
      try {
        const query = event.messages?.filter((m: any) => m.role === "user")
          .map((m: any) => typeof m.content === "string" ? m.content : 
            m.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("\n"))
          .join("\n");
        
        if (!query) return;
        
        const agentId = event.agentId || "default";
        const isolationMode = config.isolationMode || "none";
        
        const filter: any = {};
        if (isolationMode.includes("agent") || isolationMode === "agent") {
          filter.agentId = agentId;
        }
        if (isolationMode.includes("user") || isolationMode === "user") {
          filter.userId = event.userId;
        }
        if (isolationMode.includes("session") || isolationMode === "session") {
          filter.sessionId = event.sessionId;
        }
        
        const result = await api.tools.get("memu_retrieve").execute({
          query_text: query,
          limit: 3,
          filter
        });
        
        if (result.success && result.data.memories.length > 0) {
          event.context = event.context || {};
          event.context.memories = result.data.memories;
          console.log("✅ Proactively loaded", result.data.memories.length, "memories");
        }
      } catch (error) {
        console.error("Proactive retrieval error:", error);
      }
    });
  }

  // 激活
  initServices().catch(console.error);
}

export default register;
