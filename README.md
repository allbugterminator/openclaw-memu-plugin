# memU Plugin for OpenClaw

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13+-green.svg)](https://www.python.org/downloads/)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-0.1.0+-orange.svg)](https://openclaw.ai)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6.svg)](https://www.typescriptlang.org/)

[24/7 Proactive Memory](https://github.com/NevaMind-AI/memU) integration for OpenClaw AI agents. Give your AI assistant permanent memory that learns from every conversation.

## Why memU?

Traditional AI assistants have no memory - they forget everything after each conversation. memU changes this by providing:

- **Persistent Memory**: Remembers facts, preferences, and skills across sessions
- **Proactive Context**: Surfaces relevant memories before you even ask
- **Cost Efficient**: Reduces token costs with smart context caching (~1/10 of comparable usage)
- **Hierarchical Storage**: Organized like a file system - categories, items, and resources

## Features

- **Continuous Learning**: Automatically memorize facts, preferences, and skills from conversations
- **Proactive Retrieval**: Context-aware memory surfacing before responding to queries
- **Multiple Storage Backends**: In-memory or PostgreSQL (with pgvector)
- **Flexible LLM Providers**: OpenAI, OpenRouter, or custom endpoints
- **Cloud or Self-Hosted**: Use memU Cloud API or deploy your own

## Quick Start

### 1. Install Dependencies

```bash
# Install memU Python package
pip install memu-py

# Optional: For PostgreSQL storage
# docker run -d --name memu-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=memu -p 5432:5432 pgvector/pgvector:pg16

# Required for PostgreSQL storage (psycopg2 driver)
# pip install psycopg2-binary

# Required for external embedding API (node-fetch)
npm install node-fetch
```

### 2. Install the Plugin

```bash
# Clone this repository
git clone https://github.com/allbugterminator/openclaw-memu-plugin.git

# Copy to OpenClaw extensions directory
cp -r openclaw-memu-plugin ~/.openclaw/extensions/memu

# Build the plugin
cd ~/.openclaw/extensions/memu
npm install
npm run build

# Restart the Gateway
openclaw gateway restart
```

### 3. Configure

Add to your OpenClaw `openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "provider": "self-hosted",
          "storageType": "postgres",
          "postgresConnectionString": "postgresql://postgres:postgres@localhost:5432/memu",
          "llmProvider": "openai",
          "llmApiKey": "your-openai-api-key",
          "llmModel": "gpt-4o",
          "embeddingModel": "text-embedding-3-small",
          "embeddingProvider": "openai",
          "embeddingApiKey": "your-embedding-api-key",
          "embeddingBaseUrl": "https://api.openai.com/v1",
          "autoLearn": true,
          "proactiveRetrieval": true,
          "isolationMode": "none"
        }
      }
    }
  }
}
```

#### Hook Configuration

The plugin uses **lifecycle hooks** to automatically manage memory. These hooks are enabled by default via `autoLearn` and `proactiveRetrieval` settings:

| Config Option | Hook | Description | Default |
|---------------|------|-------------|---------|
| `autoLearn` | `agent_end` | Automatically memorize conversations after each agent run | `true` |
| `proactiveRetrieval` | `before_agent_start` | Retrieve relevant memories before each agent run | `true` |

**Minimal configuration with hooks enabled:**
```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "storageType": "postgres",
          "postgresConnectionString": "postgresql://postgres:postgres@localhost:5432/memu",
          "embeddingApiKey": "your-api-key",
          "embeddingBaseUrl": "https://api.openai.com/v1",
          "autoLearn": true,
          "proactiveRetrieval": true
        }
      }
    }
  }
}
```

**Disable hooks (manual mode):**
```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "autoLearn": false,
          "proactiveRetrieval": false
        }
      }
    }
  }
}
```
When both hooks are disabled, you must manually call `memu_memorize` and `memu_retrieve` tools.

## Complete Deployment Guide

### Prerequisites

- Node.js 18+ and npm
- Python 3.13+ (optional, for local memU service)
- PostgreSQL 16+ with pgvector extension (recommended for production)
- OpenAI API key or compatible embedding API

### Step 1: Setup PostgreSQL with pgvector

```bash
# Run PostgreSQL with pgvector in Docker
docker run -d \
  --name memu-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=memu \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# Verify pgvector extension is available
docker exec memu-postgres psql -U postgres -d memu -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 2: Build and Install Plugin

```bash
# Navigate to plugin directory
cd ~/.openclaw/extensions/memu

# Install dependencies
npm install

# Build TypeScript
npm run build

# Verify build output
ls -la dist/
```

### Step 3: Configure OpenClaw

Edit `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "provider": "self-hosted",
          "storageType": "postgres",
          "postgresConnectionString": "postgresql://postgres:postgres@localhost:5432/memu",
          "llmProvider": "openai",
          "llmApiKey": "sk-your-openai-api-key",
          "llmModel": "gpt-4o",
          "embeddingProvider": "openai",
          "embeddingApiKey": "sk-your-embedding-api-key",
          "embeddingBaseUrl": "https://api.openai.com/v1",
          "embeddingModel": "text-embedding-3-small",
          "embeddingDimensions": 1536,
          "autoLearn": true,
          "proactiveRetrieval": true,
          "isolationMode": "none",
          "maxContextMemories": 3,
          "similarityThreshold": 0.7
        }
      }
    }
  }
}
```

### Step 4: Restart Gateway

```bash
# Restart OpenClaw gateway
openclaw gateway restart

# Or if using systemd
sudo systemctl restart openclaw-gateway

# Verify plugin is loaded
openclaw rpc call memu.status
```

### Step 5: Verify Installation

```bash
# Check plugin status
openclaw rpc call memu.status

# Test memory storage
openclaw rpc call memu.memorize --params '{"text": "Test memory", "metadata": {"type": "test"}}'

# Test memory retrieval
openclaw rpc call memu.retrieve --params '{"query_text": "test"}'

# Check PostgreSQL records
docker exec memu-postgres psql -U postgres -d memu -c "SELECT COUNT(*) FROM memories;"
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | `cloud` \| `self-hosted` | `self-hosted` | Use memU Cloud or self-hosted |
| `cloudApiKey` | string | - | memU Cloud API key |
| `storageType` | `inmemory` \| `postgres` | `postgres` | Storage backend |
| `postgresConnectionString` | string | - | PostgreSQL connection string |
| `llmProvider` | `openai` \| `openrouter` \| `custom` | `openai` | LLM provider |
| `llmApiKey` | string | - | LLM API key |
| `llmBaseUrl` | string | - | Custom LLM base URL |
| `llmModel` | string | `gpt-4o` | LLM model |
| `embeddingProvider` | `openai` \| `custom` | `openai` | Embedding provider |
| `embeddingApiKey` | string | - | Embedding API key |
| `embeddingBaseUrl` | string | `https://api.openai.com/v1` | Embedding API base URL |
| `embeddingModel` | string | `text-embedding-3-small` | Embedding model |
| `embeddingDimensions` | number | `1536` | Embedding vector dimensions |
| `autoLearn` | boolean | `true` | Auto-memorize conversations via agent_end hook |
| `proactiveRetrieval` | boolean | `true` | Enable proactive retrieval via before_agent_start hook |
| `isolationMode` | `none` \| `agent` \| `user` \| `session` | `none` | Memory isolation mode |
| `maxContextMemories` | number | `3` | Maximum memories to inject into context |
| `similarityThreshold` | number | `0.7` | Minimum similarity score for retrieval |

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | `cloud` \| `self-hosted` | `cloud` | Use memU Cloud or self-hosted |
| `cloudApiKey` | string | - | memU Cloud API key |
| `storageType` | `inmemory` \| `postgres` | `inmemory` | Storage backend |
| `postgresConnectionString` | string | - | PostgreSQL connection string |
| `llmProvider` | `openai` \| `openrouter` \| `custom` | `openai` | LLM provider |
| `llmApiKey` | string | - | LLM API key |
| `llmBaseUrl` | string | - | Custom LLM base URL |
| `llmModel` | string | `gpt-4o` | LLM model |
| `embeddingModel` | string | `text-embedding-3-small` | Embedding model |
| `autoLearn` | boolean | `true` | Auto-memorize conversations |
| `proactiveRetrieval` | boolean | `true` | Enable proactive context |

## Available Tools

### memu_memorize

Store information in memU memory. Use this to remember facts, preferences, skills, and important context.

```typescript
// Memorize a user preference
await memu_memorize({
  content: "User prefers to be addressed in a formal manner",
  modality: "conversation"
});

// Memorize from a document
await memu_memorize({
  content: "Python best practices: use type hints, write docstrings, follow PEP 8",
  modality: "document"
});
```

### memu_retrieve

Retrieve relevant memories for context. Supports two retrieval methods:

- **`rag`**: Fast embedding-based retrieval (recommended for most cases)
- **`llm`**: Deep reasoning-based retrieval (slower but more accurate for complex queries)

```typescript
// Get context before responding
const result = await memu_retrieve({
  query_text: "What are user's communication preferences?",
  method: "rag"
});

// Deep reasoning retrieval
const deepResult = await memu_retrieve({
  query_text: "What should I know about this user?",
  method: "llm"
});
```

### memu_search

Quick search for specific facts in memory.

```typescript
// Quick search
const searchResult = await memu_search({
  query: "user's programming language preferences"
});
```

## Gateway RPC Methods

```bash
# Get plugin status
openclaw rpc call memu.status

# Check plugin health
openclaw rpc call memu.health
```

## Usage Examples

### Example 1: Remembering User Preferences

```
User: I prefer receiving weekly summary emails on Fridays.
Agent: I'll remember that you prefer weekly summary emails on Fridays. 
       Would you like me to set up any automation for this?
```

The agent can then call `memu_memorize` to store this preference for future reference.

### Example 2: Context-Aware Responses

When the user asks "What did I work on last week?", the agent:

1. Calls `memu_retrieve` with the query
2. Gets relevant memories about past projects
3. Provides a personalized, context-aware response

### Example 3: Skill Learning

The agent observes user behavior and learns skills:

```
User: [Uses vim keybindings throughout the session]
Agent: [memorizes user's preference for vim keybindings]
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   OpenClaw      │     │    memU Plugin  │     │  PostgreSQL     │
│   Agent         │────►│                 │────►│  + pgvector     │
│                 │     │  ┌───────────┐  │     │                 │
└─────────────────┘     │  │  Hooks    │  │     └─────────────────┘
        │               │  │ • agent_end│  │              │
        │               │  │ • before_ │  │              │
        │               │  │   agent_start│ │              │
        │               │  └───────────┘  │              │
        │               │  ┌───────────┐  │              │
        │               │  │   Tools   │  │              │
        │               │  │•memorize  │  │              │
        │               │  │•retrieve  │  │              │
        │               │  │•search    │  │              │
        │               │  └───────────┘  │              │
        │               │  ┌───────────┐  │              │
        │               │  │ Embedding │  │              │
        │               │  │  Service  │◄─┘              │
        │               │  │(External  │                 │
        │               │  │ API)       │                 │
        │               │  └───────────┘                 │
        │               └─────────────────┘              │
        │                                                │
        │ Tools:                                         │
        │ - memu_memorize ──► Store in PostgreSQL        │
        │ - memu_retrieve ──► Query with vector search   │
        │ - memu_search ────► Full-text search           │
```

### How It Works

1. **Automatic Learning** (`agent_end` hook):
   - Triggered after each agent run completes
   - Automatically stores conversation to PostgreSQL
   - Generates embedding vector via external API

2. **Proactive Retrieval** (`before_agent_start` hook):
   - Triggered before each agent run starts
   - Retrieves relevant memories based on user query
   - Injects memories into `event.prependContext`

3. **Vector Storage**:
   - Uses PostgreSQL with pgvector extension
   - Stores text content + embedding vector + metadata
   - Supports similarity search with cosine distance

4. **Embedding Generation**:
   - Calls external API (OpenAI, 88api.chat, etc.)
   - Falls back to local pseudo-embedding on failure
   - Configurable model and dimensions

## Hooks (Lifecycle Events)

The plugin registers two lifecycle hooks that enable automatic memory management:

### 1. `agent_end` Hook - Automatic Learning

Triggered after each agent run completes. Automatically extracts and stores conversation pairs.

**Event Data:**
```typescript
{
  success: boolean;        // Whether the agent run succeeded
  messages: Message[];     // Full conversation history
  result?: any;           // Agent execution result
}
```

**Context Data:**
```typescript
{
  agentId: string;              // Current agent ID
  sessionId: string;            // Session identifier
  requesterSenderId: string;    // User identifier
}
```

**Behavior:**
- Extracts user query + assistant response pairs from `messages`
- Generates embedding vector via external API
- Stores to PostgreSQL with metadata (type, timestamp, isolation fields)
- Supports isolation modes: `none`, `agent`, `user`, `session`

**Example stored memory:**
```json
{
  "content": "用户: 我喜欢用Python编程\n助手: 好的，我会记住你喜欢Python编程",
  "metadata": {
    "type": "conversation",
    "timestamp": 1709836800000,
    "agentId": "main",
    "userId": "user123"
  }
}
```

### 2. `before_agent_start` Hook - Proactive Retrieval

Triggered before each agent run starts. Automatically retrieves relevant memories and injects them into context.

**Event Data:**
```typescript
{
  messages: Message[];        // Incoming messages
  prependContext?: string;    // Context to prepend (modified by hook)
}
```

**Behavior:**
- Extracts user query from `messages` (filters for role="user")
- Performs vector similarity search with configured filters
- Injects retrieved memories into `event.prependContext`
- Maximum 3 memories retrieved by default (configurable via `maxContextMemories`)

**Example injected context:**
```
相关记忆:
用户: 我喜欢用Python编程
助手: 好的，我会记住你喜欢Python编程

用户: 我的邮箱是example@email.com
助手: 已记录你的邮箱地址
```

### Hook Configuration

Enable/disable hooks via configuration:

```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "autoLearn": true,           // Enable agent_end hook
          "proactiveRetrieval": true,  // Enable before_agent_start hook
          "isolationMode": "agent"     // Memory isolation: none | agent | user | session
        }
      }
    }
  }
}
```

### Isolation Modes

Control memory visibility across different contexts:

| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | All memories shared globally | Single-user personal assistant |
| `agent` | Memories isolated by agent ID | Multi-agent system |
| `user` | Memories isolated by user ID | Multi-user chatbot |
| `session` | Memories isolated by session | Temporary/ephemeral contexts |

**Metadata fields added based on isolation mode:**
- `agent` mode: adds `agentId` to memory metadata and filter
- `user` mode: adds `userId` to memory metadata and filter  
- `session` mode: adds `sessionId` to memory metadata and filter

### Debugging Hooks

Enable verbose logging to see hook execution:

```bash
# Check gateway logs for hook activity
tail -f /tmp/openclaw/openclaw-*.log | grep -E "\[memu\]"

# Expected output:
# [memu] before_agent_start hook triggered
# [memu] Proactive retrieval for query: ...
# ✅ [memu] Proactively loaded 2 memories
# [memu] agent_end hook triggered
# [memu] Auto-learning conversation: ...
```

## External Embedding API Configuration

To use external embedding API (e.g., 88api.chat, OpenAI compatible):

```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "provider": "self-hosted",
          "storageType": "postgres",
          "postgresConnectionString": "postgresql://postgres:postgres@localhost:5432/memu",
          "embeddingProvider": "custom",
          "embeddingApiKey": "your-api-key",
          "embeddingBaseUrl": "https://api.88api.chat/v1",
          "embeddingModel": "text-embedding-3-small",
          "embeddingDimensions": 1536
        }
      }
    }
  }
}
```

The plugin will:
1. Call external embedding API for vector generation
2. Fall back to local pseudo-embedding if API fails
3. Store vectors in PostgreSQL with pgvector

## Cloud API Configuration

To use memU Cloud instead of self-hosted:

```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "provider": "cloud",
          "cloudApiKey": "your-memu-cloud-api-key"
        }
      }
    }
  }
}
```

Get your API key at [memu.so](https://memu.so).

## Requirements

- Python 3.13+ (for self-hosted memU)
- memU Python package: `pip install memu-py`
- PostgreSQL driver: `pip install psycopg2-binary` (required for PostgreSQL storage)
- For PostgreSQL storage: PostgreSQL with pgvector extension

## Troubleshooting

### Plugin not loading

```bash
# Check if plugin is loaded
openclaw plugins list

# View gateway logs
tail -f /tmp/openclaw/openclaw-*.log | grep -i memu

# Check for TypeScript compilation errors
cd ~/.openclaw/extensions/memu
npm run build
```

### PostgreSQL connection errors

```bash
# Verify PostgreSQL is running
docker ps | grep memu-postgres

# Test connection
docker exec memu-postgres psql -U postgres -d memu -c "SELECT 1;"

# Check pgvector extension
docker exec memu-postgres psql -U postgres -d memu -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Embedding API errors

If you see `TypeError: Cannot read properties of undefined` or embedding failures:

```bash
# Check embedding API configuration in openclaw.json
# Verify embeddingApiKey and embeddingBaseUrl are correct

# Test embedding API directly
curl -X POST https://api.88api.chat/v1/embeddings \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-3-small"}'
```

### Hooks not triggering

If `agent_end` or `before_agent_start` hooks are not working:

```bash
# Check gateway logs for hook registration
tail -f /tmp/openclaw/openclaw-*.log | grep -E "(hook|agent_end|before_agent_start)"

# Verify plugin is using correct hook API (api.on, not api.registerHook)
# Check logs for [memu] prefix messages
```

### Python not found

Make sure Python 3.13+ is installed and available in PATH:
```bash
python --version
```

### memu-py not installed

```bash
pip install memu-py
```

### Import errors

If you see import errors, ensure memu-py is correctly installed:
```bash
python -c "from memu.app import MemoryService; print('OK')"
```

### TypeScript compilation errors

```bash
# Clean and rebuild
cd ~/.openclaw/extensions/memu
rm -rf dist/
npm install
npm run build

# Check for type errors
npx tsc --noEmit
```

## Related Projects

- [memU](https://github.com/NevaMind-AI/memU) - Core proactive memory engine
- [memUBot](https://github.com/NevaMind-AI/memUBot) - Enterprise-ready OpenClaw with memU
- [OpenClaw](https://github.com/openclaw/openclaw) - Open source AI coding assistant

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

If you find this plugin useful, please consider starring the [memU repository](https://github.com/NevaMind-AI/memU)!
