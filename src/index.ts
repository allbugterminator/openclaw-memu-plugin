import { randomUUID } from "crypto";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import pg from "pg";

const { Pool } = pg;

// Extension Context Types
type ExtensionConfig = Record<string, any>;
interface ExtensionContext {
  config: ExtensionConfig;
  logger: {
    info: (...args: any[]) => void;
    error: (...args: any[]) => void;
    debug: (...args: any[]) => void;
  };
}

// Document Type - using simple object instead of class
function createDocument(options: { pageContent: string; metadata?: Record<string, any> }) {
  return {
    pageContent: options.pageContent,
    metadata: options.metadata || {}
  };
}

// Simple keyword-based similarity (fallback when no embeddings)
function simpleKeywordSimilarity(query: string, content: string): number {
  const queryWords = query.toLowerCase().split(/\s+/);
  const contentLower = content.toLowerCase();
  let matches = 0;
  for (const word of queryWords) {
    if (contentLower.includes(word)) {
      matches++;
    }
  }
  return matches / Math.max(queryWords.length, 1);
}

// Memory Vector Store (In-Memory Fallback) - using factory function
function createMemoryVectorStore(embeddings: any) {
  const documents: any[] = [];
  const vectors: number[][] = [];

  return {
    documents,
    vectors,

    async addDocuments(docs: any[]): Promise<string[]> {
      const ids: string[] = [];
      for (const doc of docs) {
        // Generate fake embedding if no embeddings available
        if (embeddings) {
          const embedding = await embeddings.embedQuery(doc.pageContent);
          vectors.push(embedding);
        } else {
          // Use simple hash as placeholder vector
          const hash = doc.pageContent.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
          vectors.push(Array(1536).fill(0).map((_, i) => Math.sin(hash + i)));
        }
        documents.push(doc);
        ids.push(doc.metadata.id || randomUUID());
      }
      return ids;
    },

    async similaritySearchWithScore(
      query: string,
      k: number = 4,
      filter?: any
    ): Promise<[any, number][]> {
      let scores;
      
      if (embeddings) {
        const queryEmbedding = await embeddings.embedQuery(query);
        scores = vectors.map((vector, idx) => {
          const doc = documents[idx];
          if (filter?.userId && doc.metadata.userId !== filter.userId) {
            return { idx, score: -1 };
          }
          const similarity = cosineSimilarity(queryEmbedding, vector);
          return { idx, score: similarity };
        });
      } else {
        // Fallback to keyword matching
        scores = documents.map((doc, idx) => {
          if (filter?.userId && doc.metadata.userId !== filter.userId) {
            return { idx, score: -1 };
          }
          const similarity = simpleKeywordSimilarity(query, doc.pageContent);
          return { idx, score: similarity };
        });
      }
      
      const topK = scores
        .filter((s) => s.score >= 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
      
      return topK.map(({ idx, score }) => [documents[idx], score]);
    }
  };
}

// Helper function
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Extension state holder (plain object, not class)
interface ExtensionState {
  config: any;
  context: any;
  vectorStore: any;
  embeddings: any;
  pgPool: any;
  initialized: boolean;
}

// Factory function to create extension state
function createExtension(context: any) {
  const state: ExtensionState = {
    config: {
      provider: "self-hosted",
      storageType: "postgres",
      llmProvider: "custom",
      llmModel: "gpt-4o-mini",
      embeddingModel: "text-embedding-3-small",
      chunk_size: 1500,
      chunk_overlap: 200,
      ...context.config,
    },
    context,
    vectorStore: null,
    embeddings: null,
    pgPool: null,
    initialized: false
  };
  return state;
}

// Initialize services
async function initServices(state: ExtensionState) {
  if (state.initialized) return;

  // Initialize embeddings (optional - fallback to simple text matching)
  if (!state.embeddings && state.config.llmApiKey) {
    try {
      state.embeddings = new OpenAIEmbeddings({
        apiKey: state.config.llmApiKey,
        model: state.config.embeddingModel,
        configuration: {
          baseURL: state.config.llmBaseUrl,
        },
        dimensions: 1536,
      });
    } catch (error) {
      console.warn("Failed to initialize embeddings:", error);
    }
  }

  // Always initialize vector store (in-memory fallback works with or without embeddings)
  if (!state.vectorStore) {
    if (state.config.storageType === "postgres" && state.config.postgresConnectionString) {
      try {
        state.pgPool = new Pool({
          connectionString: state.config.postgresConnectionString,
        });

        const client = await state.pgPool.connect();
        
        // 启用pgvector扩展
        await client.query('CREATE EXTENSION IF NOT EXISTS vector');
        
        // 删除旧表（如果存在）让PGVectorStore重新创建正确的结构
        await client.query('DROP TABLE IF EXISTS memories CASCADE');
        
        client.release();
        
        // Now initialize PGVectorStore with existing table
        const pgConfig = {
          postgresConnectionOptions: {
            connectionString: state.config.postgresConnectionString,
          },
          tableName: "memories",
          columns: {
            idColumnName: "id",
            vectorColumnName: "embedding",
            contentColumnName: "content",
            metadataColumnName: "metadata",
          },
        };
        
        // 确保表有正确的列名映射
        // PGVectorStore需要通过ID列来跟踪记录
        state.vectorStore = await PGVectorStore.initialize(state.embeddings, {
          ...pgConfig,
        });
        
        console.log("✅ PostgreSQL with pgvector initialized successfully");
      } catch (error) {
        console.error("❌ Failed to connect to PostgreSQL, falling back to in-memory:", error);
        state.vectorStore = createMemoryVectorStore(state.embeddings);
      }
    } else {
      state.vectorStore = createMemoryVectorStore(state.embeddings);
      console.log("ℹ️ Using in-memory storage");
    }
  }

  state.initialized = true;
}

// Core methods - bound to state
async function memorize(this: any, text: string, modality: string = "conversation", userId?: string, metadata: Record<string, any> = {}): Promise<any> {
  try {
    await initServices(this);

    const id = randomUUID();
    const doc = createDocument({
      pageContent: text,
      metadata: {
        id,
        modality,
        userId,
        timestamp: Date.now(),
        ...metadata,
      },
    });

    // 传递ids选项，PGVectorStore需要这个
    await this.vectorStore.addDocuments([doc], { ids: [id] });

    return {
      success: true,
      data: { memoryId: id },
    };
  } catch (error) {
    return {
      success: false,
      error: (error as Error).message,
    };
  }
}

async function retrieve(this: any, queryText: string, method: string = "rag", userId?: string, limit: number = 5): Promise<any> {
  try {
    await initServices(this);

    const results = await this.vectorStore.similaritySearchWithScore(
      queryText,
      limit,
      userId ? { userId } : undefined
    );

    const memories = results.map(([doc, score]: [any, number]) => ({
      id: doc.metadata.id,
      content: doc.pageContent,
      similarity: score,
    }));

    return {
      success: true,
      data: { memories },
    };
  } catch (error) {
    return {
      success: false,
      error: (error as Error).message,
    };
  }
}

async function search(this: any, query: string): Promise<any> {
  return this.retrieve(query, "search", undefined, 10);
}

// Register function - entry point for OpenClaw
function register(api: any) {
  const context = {
    config: api.config || {},
    logger: api.logger || console,
  };

  const state = createExtension(context);

  // Bind methods to state
  const boundMemorize = memorize.bind(state);
  const boundRetrieve = retrieve.bind(state);
  const boundSearch = search.bind(state);

  // Activate the extension
  async function activate() {
    try {
      await initServices(state);
      console.log("✅ memU extension activated successfully");
    } catch (error) {
      console.error("❌ Failed to activate memU extension:", error);
      throw error;
    }
  }

  // Register tools
  api.registerTool({
    name: "memu_memorize",
    description: "Store a memory",
    parameters: {
      type: "object",
      required: ["text"],
      properties: {
        text: { type: "string" },
        metadata: { type: "object" }
      }
    },
    execute: ({ text, metadata }: any) => boundMemorize(text, "conversation", undefined, metadata)
  });

  api.registerTool({
    name: "memu_retrieve",
    description: "Retrieve memories",
    parameters: {
      type: "object",
      required: ["query_text"],
      properties: {
        query_text: { type: "string" },
        limit: { type: "number" }
      }
    },
    execute: ({ query_text, limit }: any) => boundRetrieve(query_text, "rag", undefined, limit || 5)
  });

  api.registerTool({
    name: "memu_search",
    description: "Search memories",
    parameters: {
      type: "object",
      required: ["query"],
      properties: {
        query: { type: "string" }
      }
    },
    execute: ({ query }: any) => boundSearch(query)
  });

  // Expose methods on state for direct access
  state.memorize = boundMemorize;
  state.retrieve = boundRetrieve;
  state.search = boundSearch;

  // Auto-learn hook: store conversation pairs when agent ends
  if (state.config.autoLearn) {
    api.on("agent_end", async (event: any) => {
      if (!event.success || !event.messages || event.messages.length === 0) {
        return;
      }

      try {
        let lastUserQuery = "";
        
        for (const msg of event.messages) {
          if (!msg || typeof msg !== "object") continue;
          
          const role = msg.role;
          let content = "";
          
          // Extract text content
          if (typeof msg.content === "string") {
            content = msg.content;
          } else if (Array.isArray(msg.content)) {
            content = msg.content
              .filter((block: any) => block?.type === "text" && block?.text)
              .map((block: any) => block.text)
              .join("\n");
          }
          
          if (!content.trim()) continue;
          
          // Pair user query with assistant response
          if (role === "user") {
            lastUserQuery = content.trim();
          } else if (role === "assistant" && lastUserQuery) {
            const conversationText = `用户查询: ${lastUserQuery}\n助手回复: ${content.trim()}`;
            await boundMemorize(conversationText, "conversation", undefined, {
              type: "conversation_pair",
              timestamp: Date.now()
            });
            lastUserQuery = ""; // Reset after pairing
          }
        }
      } catch (error) {
        console.error("❌ Auto-learn failed:", error);
      }
    });
  }

  // Activate
  activate();
  
  return state;
}

// ES Module exports
export { register };
export default register;