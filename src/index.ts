import { Extension } from "@openclaw/sdk";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { randomUUID } from "crypto";
import { Pool } from "pg";
import pgvector from "pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { Document } from "@langchain/core/documents";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatVolcengine } from "@langchain/community/chat_models/volcengine";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface MemUConfig {
  provider: "cloud" | "self-hosted";
  cloudApiKey?: string;
  cloudEndpoint?: string;
  storageType?: "inmemory" | "postgres" | "file";
  postgresConnectionString?: string;
  dataDir?: string;
  llmProvider?: "openai" | "anthropic" | "volcengine" | "custom";
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
  metadata: Record<string, any>;
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
  private workDir: string;
  private vectorStore: MemoryVectorStore | PGVectorStore | null = null;
  private embeddings: OpenAIEmbeddings | null = null;
  private llm: any = null;
  private pgPool: Pool | null = null;
  private memories: Map<string, MemoryRecord> = new Map();

  constructor(config: Partial<MemUConfig> = {}) {
    this.config = this.loadConfig(config);
    this.workDir = this.config.dataDir || path.join(homedir(), ".openclaw", "memu-data");
    this.initWorkDir();
    this.initServices();
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

    // Load existing memories from file if using file storage
    if (this.config.storageType === "file") {
      const memoryFile = path.join(this.workDir, "memories.json");
      if (existsSync(memoryFile)) {
        try {
          const data = JSON.parse(readFileSync(memoryFile, "utf-8"));
          this.memories = new Map(Object.entries(data));
        } catch (e) {
          console.warn("Failed to load memories from file:", e);
        }
      }
    }
  }

  private async initServices(): Promise<void> {
    // Initialize embeddings
    if (this.config.embeddingApiKey) {
      this.embeddings = new OpenAIEmbeddings({
        openAIApiKey: this.config.embeddingApiKey,
        model: this.config.embeddingModel || "text-embedding-ada-002",
        configuration: {
          baseURL: this.config.embeddingBaseUrl,
        },
      });
    } else {
      // Fallback: use simple hash-based embedding for demo
      this.embeddings = {
        embedQuery: async (text: string) => this.simpleEmbed(text),
        embedDocuments: async (texts: string[]) => Promise.all(texts.map(t => this.simpleEmbed(t))),
      } as any;
    }

    // Initialize LLM
    if (this.config.llmApiKey) {
      switch (this.config.llmProvider) {
        case "openai":
          this.llm = new ChatOpenAI({
            openAIApiKey: this.config.llmApiKey,
            model: this.config.llmModel || "gpt-3.5-turbo",
            configuration: {
              baseURL: this.config.llmBaseUrl,
            },
            temperature: 0.1,
          });
          break;
        case "anthropic":
          this.llm = new ChatAnthropic({
            anthropicApiKey: this.config.llmApiKey,
            model: this.config.llmModel || "claude-3-sonnet-20240229",
            anthropicBaseUrl: this.config.llmBaseUrl,
            temperature: 0.1,
          });
          break;
        case "volcengine":
          this.llm = new ChatVolcengine({
            apiKey: this.config.llmApiKey,
            model: this.config.llmModel || "doubao-lite-4k",
            baseURL: this.config.llmBaseUrl,
            temperature: 0.1,
          });
          break;
      }
    }

    // Initialize vector store
    switch (this.config.storageType) {
      case "inmemory":
        this.vectorStore = new MemoryVectorStore(this.embeddings!);
        break;
      case "postgres":
        if (!this.config.postgresConnectionString) {
          throw new Error("PostgreSQL connection string is required for postgres storage");
        }
        
        // Initialize pgvector
        const pgConfig = {
          connectionString: this.config.postgresConnectionString,
        };
        
        this.pgPool = new Pool(pgConfig);
        
        // Create table if not exists
        await this.pgPool.query(`
          CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            modality TEXT NOT NULL,
            user_id TEXT,
            session_id TEXT,
            timestamp BIGINT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            embedding vector(1536)
          );
          CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
          CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
          CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        `);

        this.vectorStore = await PGVectorStore.initialize(this.embeddings!, {
          pool: this.pgPool,
          tableName: "memories",
          columns: {
            idColumnName: "id",
            contentColumnName: "content",
            metadataColumnName: "metadata",
            vectorColumnName: "embedding",
          },
        });
        break;
      case "file":
        // Use memory vector store and persist to file
        this.vectorStore = new MemoryVectorStore(this.embeddings!);
        // Load existing documents into vector store
        const docs = Array.from(this.memories.values()).map(m => new Document({
          pageContent: m.content,
          metadata: {
            id: m.id,
            modality: m.modality,
            userId: m.userId,
            timestamp: m.timestamp,
            ...m.metadata,
          },
        }));
        if (docs.length > 0) {
          await this.vectorStore.addDocuments(docs);
        }
        break;
    }
  }

  private async simpleEmbed(text: string): Promise<number[]> {
    // Simple hash-based embedding for fallback (1536 dimensions)
    const hash = this.simpleHash(text);
    const embedding = new Array(1536).fill(0);
    for (let i = 0; i < 1536; i++) {
      embedding[i] = (hash * (i + 1)) % 2 - 1;
    }
    // Normalize
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    return embedding.map(v => v / norm);
  }

  private simpleHash(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  private async persistToFile(): Promise<void> {
    if (this.config.storageType === "file") {
      const memoryFile = path.join(this.workDir, "memories.json");
      const data = Object.fromEntries(this.memories.entries());
      writeFileSync(memoryFile, JSON.stringify(data, null, 2), "utf-8");
    }
  }

  // Core functionality - PURE TS IMPLEMENTATION
  async memorize(content: string, modality: string = "conversation", userId?: string, metadata: Record<string, any> = {}): Promise<MemUResult> {
    try {
      if (!this.vectorStore) {
        await this.initServices();
      }

      const id = randomUUID();
      const timestamp = Date.now();
      
      const memory: MemoryRecord = {
        id,
        content,
        modality: modality as any,
        userId,
        timestamp,
        metadata,
      };

      // Add to vector store
      const doc = new Document({
        pageContent: content,
        metadata: {
          id,
          modality,
          userId,
          timestamp,
          ...metadata,
        },
      });

      await this.vectorStore!.addDocuments([doc]);

      // Store in memory map
      this.memories.set(id, memory);

      // Persist to file if needed
      await this.persistToFile();

      // Also persist to PostgreSQL if using postgres storage
      if (this.config.storageType === "postgres" && this.pgPool) {
        const embedding = await this.embeddings!.embedQuery(content);
        await this.pgPool.query(
          `INSERT INTO memories (id, content, modality, user_id, timestamp, metadata, embedding)
           VALUES ($1, $2, $3, $4, $5, $6, $7)
           ON CONFLICT (id) DO UPDATE SET
             content = EXCLUDED.content,
             modality = EXCLUDED.modality,
             user_id = EXCLUDED.user_id,
             timestamp = EXCLUDED.timestamp,
             metadata = EXCLUDED.metadata,
             embedding = EXCLUDED.embedding`,
          [id, content, modality, userId, timestamp, JSON.stringify(metadata), pgvector.toSql(embedding)]
        );
      }

      return {
        success: true,
        data: {
          memoryId: id,
          timestamp,
        },
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to memorize",
      };
    }
  }

  async retrieve(queries: Array<{ role: string; content: { text: string } }>, method: "rag" | "llm" = "rag", userId?: string, options: RetrievalOptions = {}): Promise<MemUResult> {
    try {
      if (!this.vectorStore) {
        await this.initServices();
      }

      const topK = options.topK || this.config.retrievalTopK || 5;
      const threshold = options.threshold || this.config.retrievalThreshold || 0.7;

      // Combine all query texts
      const queryText = queries.map(q => q.content.text).join(" ");

      // Get relevant documents
      const docsWithScores = await this.vectorStore!.similaritySearchWithScore(
        queryText,
        topK,
        userId ? { userId } : options.filter
      );

      // Filter by threshold
      const filteredDocs = docsWithScores
        .filter(([_, score]) => score >= threshold)
        .map(([doc, score]) => {
          const memory = this.memories.get(doc.metadata.id) || {
            id: doc.metadata.id,
            content: doc.pageContent,
            modality: doc.metadata.modality,
            userId: doc.metadata.userId,
            timestamp: doc.metadata.timestamp,
            metadata: doc.metadata,
          };
          return {
            ...memory,
            score,
          };
        });

      // If using LLM method, rerank with LLM
      if (method === "llm" && this.llm) {
        const reranked = await this.rerankWithLLM(queryText, filteredDocs);
        return {
          success: true,
          data: reranked,
        };
      }

      return {
        success: true,
        data: filteredDocs,
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to retrieve",
      };
    }
  }

  private async rerankWithLLM(query: string, docs: MemoryRecord[]): Promise<MemoryRecord[]> {
    if (!this.llm) return docs;

    const prompt = ChatPromptTemplate.fromTemplate(`
You are a memory reranker. Given a user query and a list of memory records, rank them by relevance to the query.
Return only a JSON array of memory IDs in order of relevance (most relevant first).
Do not include any other text or explanation.

Query: {query}

Memories:
{memories}
`);

    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());

    const memoriesText = docs.map((d, i) => `
[${i}] ID: ${d.id}
Content: ${d.content}
Score: ${d.score}
`).join("\n");

    const result = await chain.invoke({
      query,
      memories: memoriesText,
    });

    try {
      const rankedIds = JSON.parse(result) as string[];
      const idToDoc = new Map(docs.map(d => [d.id, d]));
      return rankedIds.map(id => idToDoc.get(id)).filter(Boolean) as MemoryRecord[];
    } catch (e) {
      console.warn("Failed to parse LLM reranking result:", e);
      return docs;
    }
  }

  async search(query: string, userId?: string, options: RetrievalOptions = {}): Promise<MemUResult> {
    // Search is a simpler version of retrieve, optimized for speed
    return this.retrieve([{ role: "user", content: { text: query } }], "rag", userId, options);
  }

  // Advanced functionality - PURE TS IMPLEMENTATION
  async delete(memoryId: string, userId?: string): Promise<MemUResult> {
    try {
      const memory = this.memories.get(memoryId);
      if (!memory) {
        return {
          success: false,
          error: "Memory not found",
        };
      }

      if (userId && memory.userId !== userId) {
        return {
          success: false,
          error: "Permission denied",
        };
      }

      // Remove from memory map
      this.memories.delete(memoryId);

      // Remove from vector store
      if (this.vectorStore && "delete" in this.vectorStore) {
        await this.vectorStore.delete({ ids: [memoryId] });
      }

      // Remove from PostgreSQL
      if (this.config.storageType === "postgres" && this.pgPool) {
        await this.pgPool.query("DELETE FROM memories WHERE id = $1", [memoryId]);
      }

      // Persist to file
      await this.persistToFile();

      return {
        success: true,
        data: {
          deletedId: memoryId,
        },
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to delete memory",
      };
    }
  }

  async list(userId?: string, limit: number = 100, offset: number = 0): Promise<MemUResult> {
    try {
      let memories = Array.from(this.memories.values());

      // Filter by user
      if (userId) {
        memories = memories.filter(m => m.userId === userId);
      }

      // Sort by timestamp (newest first)
      memories.sort((a, b) => b.timestamp - a.timestamp);

      // Paginate
      const paginated = memories.slice(offset, offset + limit);

      return {
        success: true,
        data: {
          memories: paginated,
          total: memories.length,
          offset,
          limit,
        },
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to list memories",
      };
    }
  }

  async clear(userId?: string): Promise<MemUResult> {
    try {
      let count = 0;

      if (userId) {
        // Delete only user's memories
        const userMemories = Array.from(this.memories.values()).filter(m => m.userId === userId);
        count = userMemories.length;
        
        for (const memory of userMemories) {
          this.memories.delete(memory.id);
        }

        // Delete from vector store
        if (this.vectorStore && "delete" in this.vectorStore) {
          await this.vectorStore.delete({ ids: userMemories.map(m => m.id) });
        }

        // Delete from PostgreSQL
        if (this.config.storageType === "postgres" && this.pgPool) {
          await this.pgPool.query("DELETE FROM memories WHERE user_id = $1", [userId]);
        }
      } else {
        // Delete all memories
        count = this.memories.size;
        this.memories.clear();

        // Clear vector store
        if (this.vectorStore && "delete" in this.vectorStore) {
          // For memory vector store, recreate it
          if (this.vectorStore instanceof MemoryVectorStore) {
            this.vectorStore = new MemoryVectorStore(this.embeddings!);
          } else {
            // For PGVector, truncate table
            if (this.pgPool) {
              await this.pgPool.query("TRUNCATE TABLE memories");
            }
          }
        }
      }

      // Persist to file
      await this.persistToFile();

      return {
        success: true,
        data: {
          deletedCount: count,
        },
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to clear memories",
      };
    }
  }

  async getStats(userId?: string): Promise<MemUResult> {
    try {
      let memories = Array.from(this.memories.values());

      if (userId) {
        memories = memories.filter(m => m.userId === userId);
      }

      const modalities = new Map<string, number>();
      let earliestTimestamp = Infinity;
      let latestTimestamp = 0;

      for (const memory of memories) {
        modalities.set(memory.modality, (modalities.get(memory.modality) || 0) + 1);
        earliestTimestamp = Math.min(earliestTimestamp, memory.timestamp);
        latestTimestamp = Math.max(latestTimestamp, memory.timestamp);
      }

      return {
        success: true,
        data: {
          totalMemories: memories.length,
          modalities: Object.fromEntries(modalities),
          earliestMemory: earliestTimestamp === Infinity ? null : new Date(earliestTimestamp).toISOString(),
          latestMemory: latestTimestamp === 0 ? null : new Date(latestTimestamp).toISOString(),
          storageType: this.config.storageType,
          provider: this.config.provider,
        },
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to get stats",
      };
    }
  }

  // Batch operations - PURE TS IMPLEMENTATION
  async batchMemorize(records: Array<Omit<MemoryRecord, "id" | "timestamp">>): Promise<MemUResult> {
    try {
      const results = [];
      for (const record of records) {
        const result = await this.memorize(
          record.content,
          record.modality,
          record.userId,
          record.metadata
        );
        results.push(result);
      }

      return {
        success: results.every(r => r.success),
        data: results,
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to batch memorize",
      };
    }
  }

  async batchDelete(memoryIds: string[], userId?: string): Promise<MemUResult> {
    try {
      const results = [];
      for (const id of memoryIds) {
        const result = await this.delete(id, userId);
        results.push(result);
      }

      return {
        success: results.every(r => r.success),
        data: results,
      };
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to batch delete",
      };
    }
  }

  // Auto-learn functionality - PURE TS IMPLEMENTATION
  async autoLearnFromConversation(messages: Array<{ role: string; content: string }>, userId?: string): Promise<MemUResult> {
    if (!this.config.autoLearn) {
      return {
        success: false,
        error: "Auto-learn is disabled in config",
      };
    }

    try {
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
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Failed to auto learn",
      };
    }
  }

  private async extractImportantPoints(messages: Array<{ role: string; content: string }>): Promise<string[]> {
    if (!this.llm) {
      // Fallback: return all user messages longer than 10 chars
      return messages
        .filter(m => m.role === "user" || m.role === "assistant")
        .map(m => m.content)
        .filter(c => c.length > 10);
    }

    const prompt = ChatPromptTemplate.fromTemplate(`
Extract the most important facts, preferences, decisions, and information from the following conversation.
Return only a JSON array of strings, each being a single important point.
Do not include any other text or explanation.

Conversation:
{conversation}
`);

    const chain = prompt.pipe(this.llm).pipe(new StringOutputParser());

    const conversationText = messages.map(m => `${m.role}: ${m.content}`).join("\n");

    const result = await chain.invoke({
      conversation: conversationText,
    });

    try {
      return JSON.parse(result) as string[];
    } catch (e) {
      console.warn("Failed to parse important points:", e);
      return result.split("\n").filter(line => line.trim().length > 0);
    }
  }

  // Proactive retrieval - PURE TS IMPLEMENTATION
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

  // Cloud mode implementation - PURE TS IMPLEMENTATION
  private async cloudRequest(endpoint: string, body: any): Promise<MemUResult> {
    if (!this.config.cloudApiKey) {
      return {
        success: false,
        error: "Cloud API key is required for cloud mode",
      };
    }

    try {
      const fetch = (await import("node-fetch")).default;
      const baseUrl = this.config.cloudEndpoint || "https://api.memu.so/v1";
      
      const response = await fetch(`${baseUrl}${endpoint}`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.config.cloudApiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        timeout: 30000,
      });

      if (!response.ok) {
        return {
          success: false,
          error: `Cloud API error: ${response.status} ${response.statusText}`,
        };
      }

      return await response.json() as MemUResult;
    } catch (e: any) {
      return {
        success: false,
        error: e.message || "Cloud request failed",
      };
    }
  }
}

const extension = new Extension({
  name: "memu",
  description: "MemU proactive memory plugin for OpenClaw - 100% pure TypeScript implementation",
  version: "2.0.0",
});

let memuClient: MemUClient;

extension.on("init", async () => {
  memuClient = new MemUClient();
  extension.logger.info("MemU plugin initialized with 100% pure TypeScript implementation");

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
        description: "Retrieval method: 'rag' for fast embedding-based, 'llm' for deep reasoning reranking",
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
  description: "Quick search for specific facts or preferences in memU memory. Optimized for speed, uses RAG only.",
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
  description: "List all memories for a user with pagination, sorted by newest first.",
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
  description: "Clear all memories for a user. Use with caution! This action cannot be undone.",
  parameters: {
    type: "object",
    properties: {
      user_id: {
        type: "string",
        description: "Optional user identifier to clear memories for. If not provided, clears all memories.",
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
  description: "Get memory statistics for a user (total memories, modality breakdown, storage info, etc.)",
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
  description: "Batch store multiple memories at once for better performance.",
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
