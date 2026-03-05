# memU Plugin for OpenClaw

[24/7 Proactive Memory](https://github.com/NevaMind-AI/memU) integration for OpenClaw AI agents.

## Features

- **Continuous Learning**: Automatically memorize facts, preferences, and skills from conversations
- **Proactive Retrieval**: Context-aware memory surfacing before responding to queries
- **Multiple Storage Backends**: In-memory or PostgreSQL (with pgvector)
- **Flexible LLM Providers**: OpenAI, OpenRouter, or custom endpoints
- **Cloud or Self-Hosted**: Use memU Cloud API or deploy your own

## Installation

### Option 1: Local Development

```bash
# Clone or copy this plugin to your extensions directory
cp -r openclaw-memu-plugin ~/.openclaw/extensions/memu

# Restart the Gateway
openclaw gateway restart
```

### Option 2: Install from npm (when published)

```bash
openclaw plugins install @openclaw/memu
```

## Configuration

Add to your OpenClaw config:

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
          "embeddingModel": "text-embedding-3-small",
          "autoLearn": true,
          "proactiveRetrieval": true
        }
      }
    }
  }
}
```

### Configuration Options

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

Store information in memU memory.

```typescript
await memu_memorize({
  content: "User prefers to be addressed in a formal manner",
  modality: "conversation"
});
```

### memu_retrieve

Retrieve relevant memories for context.

```typescript
await memu_retrieve({
  queries: [
    { role: "user", content: { text: "What are user's communication preferences?" } }
  ],
  method: "rag"
});
```

### memu_search

Quick search for specific facts.

```typescript
await memu_search({
  query: "user's email preferences"
});
```

## Gateway RPC Methods

- `memu.status` - Get plugin status
- `memu.health` - Check plugin health

## Requirements

- Python 3.13+ (for self-hosted memU)
- memU Python package: `pip install memu-py`
- For PostgreSQL storage: PostgreSQL with pgvector extension

## Example Usage

### Memorize a Conversation

```
User: I prefer technical documentation with code examples
Agent: I'll remember that you prefer technical documentation with code examples.
```

The agent can then call `memu_memorize` to store this preference.

### Retrieve Context

When responding to a complex query, the agent can:
1. Call `memu_retrieve` with the user's query
2. Get relevant memories (preferences, past interactions, learned skills)
3. Provide a more personalized and context-aware response

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

## License

Apache License 2.0
