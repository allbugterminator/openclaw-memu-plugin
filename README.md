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
```

### 2. Install the Plugin

```bash
# Clone this repository
git clone https://github.com/allbugterminator/openclaw-memu-plugin.git

# Copy to OpenClaw extensions directory
cp -r openclaw-memu-plugin ~/.openclaw/extensions/memu

# Restart the Gateway
openclaw gateway restart
```

### 3. Configure

Add to your OpenClaw `config.json`:

```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "provider": "self-hosted",
          "storageType": "inmemory",
          "llmProvider": "openai",
          "llmApiKey": "your-openai-api-key",
          "llmModel": "gpt-4o",
          "embeddingModel": "text-embedding-3-small"
        }
      }
    }
  }
}
```

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
┌─────────────────┐     ┌─────────────────┐
│   OpenClaw      │     │     memU        │
│   Agent         │────►│   Memory        │
│                 │     │   Service       │
└─────────────────┘     └─────────────────┘
        │                       │
        │ Tools:                │
        │ - memu_memorize       │ Storage:
        │ - memu_retrieve       │ - In-Memory
        │ - memu_search         │ - PostgreSQL
                               │
                               │ LLM Providers:
                               │ - OpenAI
                               │ - OpenRouter
                               │ - Custom
```

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
openclaw gateway logs
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

## Related Projects

- [memU](https://github.com/NevaMind-AI/memU) - Core proactive memory engine
- [memUBot](https://github.com/NevaMind-AI/memUBot) - Enterprise-ready OpenClaw with memU
- [OpenClaw](https://github.com/openclaw/openclaw) - Open source AI coding assistant

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

If you find this plugin useful, please consider starring the [memU repository](https://github.com/NevaMind-AI/memU)!
