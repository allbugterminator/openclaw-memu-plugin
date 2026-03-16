# memU Plugin for OpenClaw

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178c6.svg)](https://www.typescriptlang.org/)

Persistent memory for OpenClaw AI agents. Automatically remembers conversations and retrieves relevant context.

## Features

- **Auto-Learn**: Remembers every conversation automatically
- **Proactive Retrieval**: Surfaces relevant memories before responding
- **Multiple Backends**: Cloud, PostgreSQL, or In-Memory
- **Hooks**: `agent_end` (auto-learn) and `before_agent_start` (proactive retrieval)

## Quick Start

### Choose Your Backend

| Backend | Setup | Persistence | Best For |
|---------|-------|-------------|----------|
| **Cloud** | 2 min | memU Cloud | Quick start |
| **PostgreSQL** | 10 min | Persistent | Production |
| **In-Memory** | 5 min | None | Development |

### Installation

```bash
git clone https://github.com/allbugterminator/openclaw-memu-plugin.git
cp -r openclaw-memu-plugin ~/.openclaw/extensions/memu
cd ~/.openclaw/extensions/memu
npm install && npm run build
openclaw gateway restart
```

### Configuration

#### Cloud
```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "provider": "cloud",
          "cloudApiKey": "your-api-key"
        }
      }
    }
  }
}
```

#### PostgreSQL
```bash
# Start PostgreSQL
docker run -d --name memu-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=memu -p 5432:5432 pgvector/pgvector:pg17
```

```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "storageType": "postgres",
          "postgresConnectionString": "postgresql://postgres:postgres@localhost:5432/memu",
          "embeddingApiKey": "your-api-key"
        }
      }
    }
  }
}
```

#### In-Memory
```json
{
  "plugins": {
    "entries": {
      "memu": {
        "enabled": true,
        "config": {
          "storageType": "inmemory",
          "embeddingApiKey": "your-api-key"
        }
      }
    }
  }
}
```

## Hooks

The plugin registers two lifecycle hooks:

### `agent_end` - Auto Learn
Triggered after each agent run. Automatically stores conversation to memory.

### `before_agent_start` - Proactive Retrieval
Triggered before each agent run. Retrieves relevant memories and injects into context.

### Hook Options

| Option | Default | Description |
|--------|---------|-------------|
| `autoLearn` | `true` | Enable `agent_end` hook |
| `proactiveRetrieval` | `true` | Enable `before_agent_start` hook |
| `isolationMode` | `none` | Memory isolation: `none`/`agent`/`user`/`session` |

Disable hooks for manual mode:
```json
{
  "autoLearn": false,
  "proactiveRetrieval": false
}
```

## Tools

### memu_memorize
Store information manually.
```typescript
await memu_memorize({
  content: "User prefers Python",
  metadata: { type: "preference" }
});
```

### memu_retrieve
Retrieve relevant memories.
```typescript
const result = await memu_retrieve({
  query_text: "What does user prefer?"
});
```

### memu_search
Quick search.
```typescript
await memu_search({ query: "Python" });
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | `cloud` \| `self-hosted` | `self-hosted` | Service provider |
| `cloudApiKey` | string | - | memU Cloud API key |
| `storageType` | `postgres` \| `inmemory` | `postgres` | Storage backend |
| `postgresConnectionString` | string | - | PostgreSQL connection string |
| `embeddingApiKey` | string | - | Embedding API key |
| `embeddingBaseUrl` | string | `https://api.openai.com/v1` | Embedding API endpoint |
| `embeddingModel` | string | `text-embedding-3-small` | Embedding model |
| `autoLearn` | boolean | `true` | Auto-memorize conversations |
| `proactiveRetrieval` | boolean | `true` | Enable proactive retrieval |
| `isolationMode` | `none` \| `agent` \| `user` \| `session` | `none` | Memory isolation |

## Troubleshooting

```bash
# Check plugin status
openclaw rpc call memu.status

# View logs
tail -f /tmp/openclaw/openclaw-*.log | grep "\[memu\]"

# Test PostgreSQL
docker exec memu-postgres psql -U postgres -d memu -c "SELECT COUNT(*) FROM memories;"
```

## License

Apache License 2.0
