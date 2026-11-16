"""The n8n wire-format constants the parser dispatches on."""

import re

N8N_CHAIN_TYPE = "@n8n/n8n-nodes-langchain.chainLlm"
N8N_CHAT_TRIGGER = "@n8n/n8n-nodes-langchain.chatTrigger"
N8N_MODEL_SPEC = "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference"
N8N_MODEL_NODE_TYPES = {
    N8N_MODEL_SPEC,
    "@n8n/n8n-nodes-langchain.lmChatAnthropic",
    "@n8n/n8n-nodes-langchain.lmChatOpenAi",
}
N8N_CODE_NODE = "n8n-nodes-base.code"
N8N_POSTGRES_NODE = "n8n-nodes-base.postgres"
N8N_S3_NODE = "n8n-nodes-base.s3"
N8N_AGENT_NODE = "@n8n/n8n-nodes-langchain.agent"

REF_PATTERN = re.compile(
    r"\{\{\s*\$\('([^']+)'\)(?:\.[^}]*)?\s*\}\}|\$\('([^']+)'\)(?:\.[\w\.\[\]]+)?"
)
REF_DETAIL_PATTERN = re.compile(r"\$\('([^']+)'\)(?:\.([\w\.\[\]]+))?")
QUERY_EXPR_PATTERN = re.compile(r"\{\{\s*(.*?)\s*\}\}")
REF_TOKEN_PREFIX = "__lumilake_ref__:"
TABLE_BLOCK_PATTERN = re.compile(r"```table\s*(.*?)```", re.IGNORECASE | re.DOTALL)
