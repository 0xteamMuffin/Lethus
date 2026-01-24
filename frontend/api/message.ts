import { apiFetch } from "./http";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface SendMessageParams {
  messages: ChatMessage[];
  userId: string;
  model?: string;
  stream?: boolean;
  conversationId?: number;  // For fetching enhanced_mode from DB
  enhancedMode?: boolean;   // Override: true = DYCP, false = passthrough
}

export interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface DYCPStats {
  enhancedMode: boolean;
  originalMessages: number;
  reducedMessages: number;
  originalTokens: number;
  reducedTokens: number;
  tokensSaved: number;
  reductionPercent: number;
  spansFound: number;
  processingMs: number;
  // Extended stats (only in enhanced mode)
  ghostEntities?: number;
  ghostBoosts?: number;
  decayLambda?: number;
  tau?: number;
  theta?: number;
  entityNames?: string[];
  spanDetails?: [number, number][];
  boostCount?: number;
}

export interface Conversation {
  id: number;
  user_id: string;
  title: string;
  enhanced_mode: boolean;
  created_at: string;
  updated_at: string;
}

export interface Turn {
  id: number;
  conversation_id: number;
  turn_number: number;
  user_message: string;
  assistant_message: string;
  timestamp: string;
  importance_score: number;
  is_pinned: boolean;
}

export interface ConversationWithTurns extends Conversation {
  turns: Turn[];
}

/**
 * Parse DYCP stats from response headers
 */
function parseDycpStats(headers: Headers): DYCPStats {
  const enhancedMode = headers.get("X-Lethus-Enhanced-Mode") === "true";
  
  const stats: DYCPStats = {
    enhancedMode,
    originalMessages: parseInt(headers.get("X-Lethus-Original-Messages") || "0"),
    reducedMessages: parseInt(headers.get("X-Lethus-Reduced-Messages") || "0"),
    originalTokens: parseInt(headers.get("X-Lethus-Original-Tokens") || "0"),
    reducedTokens: parseInt(headers.get("X-Lethus-Reduced-Tokens") || "0"),
    tokensSaved: parseInt(headers.get("X-Lethus-Tokens-Saved") || "0"),
    reductionPercent: parseFloat(headers.get("X-Lethus-Reduction-Percent") || "0"),
    spansFound: parseInt(headers.get("X-Lethus-Spans-Found") || "0"),
    processingMs: parseFloat(headers.get("X-Lethus-Processing-Ms") || "0"),
  };
  
  // Parse extended stats (only present in enhanced mode)
  if (enhancedMode) {
    stats.ghostEntities = parseInt(headers.get("X-Lethus-Ghost-Entities") || "0");
    stats.ghostBoosts = parseInt(headers.get("X-Lethus-Ghost-Boosts") || "0");
    stats.decayLambda = parseFloat(headers.get("X-Lethus-Decay-Lambda") || "0");
    stats.tau = parseFloat(headers.get("X-Lethus-Tau") || "0");
    stats.theta = parseFloat(headers.get("X-Lethus-Theta") || "0");
    stats.boostCount = parseInt(headers.get("X-Lethus-Boost-Count") || "0");
    
    // Parse JSON-encoded arrays
    try {
      const entityNames = headers.get("X-Lethus-Entity-Names");
      if (entityNames) stats.entityNames = JSON.parse(entityNames);
    } catch { /* ignore */ }
    
    try {
      const spanDetails = headers.get("X-Lethus-Span-Details");
      if (spanDetails) stats.spanDetails = JSON.parse(spanDetails);
    } catch { /* ignore */ }
  }
  
  return stats;
}

/**
 * Send chat completion request via proxy.
 * Uses user's stored API key from database.
 */
export async function sendChatCompletion(params: SendMessageParams): Promise<{
  response: ChatCompletionResponse;
  dycpStats: DYCPStats;
}> {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: params.model,
      messages: params.messages,
      user_id: params.userId,
      conversation_id: params.conversationId,
      enhanced_mode: params.enhancedMode,
      stream: false,
    }),
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Chat request failed");
  }

  const dycpStats = parseDycpStats(res.headers);
  const response = await res.json();
  return { response, dycpStats };
}

/**
 * Stream chat completion via proxy.
 * Uses user's stored API key from database.
 * Supports thinking/reasoning content from models like DeepSeek-R1.
 * Uses batched updates for smooth streaming display.
 */
export async function streamChatCompletion(
  params: SendMessageParams,
  onChunk: (content: string, fullContent: string) => void,
  onComplete?: (dycpStats: DYCPStats) => void,
  onThinking?: (thinking: string, fullThinking: string) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: params.model,
      messages: params.messages,
      user_id: params.userId,
      conversation_id: params.conversationId,
      enhanced_mode: params.enhancedMode,
      stream: true,
    }),
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Chat request failed");
  }

  // Extract DYCP stats from headers
  const dycpStats = parseDycpStats(res.headers);

  const reader = res.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error("No response body");
  }

  let buffer = "";
  
  // Accumulated content for batched updates
  let accumulatedContent = "";
  let accumulatedThinking = "";
  let pendingContentUpdate = "";
  let pendingThinkingUpdate = "";
  let rafId: number | null = null;
  
  // Flush pending updates via requestAnimationFrame for smooth rendering
  const flushUpdates = () => {
    if (pendingContentUpdate) {
      onChunk(pendingContentUpdate, accumulatedContent);
      pendingContentUpdate = "";
    }
    if (pendingThinkingUpdate && onThinking) {
      onThinking(pendingThinkingUpdate, accumulatedThinking);
      pendingThinkingUpdate = "";
    }
    rafId = null;
  };
  
  const scheduleUpdate = () => {
    if (rafId === null) {
      rafId = requestAnimationFrame(flushUpdates);
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      
      // Process complete lines from buffer
      const lines = buffer.split("\n");
      // Keep the last potentially incomplete line in buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmedLine = line.trim();
        
        if (trimmedLine === "") continue;
        if (trimmedLine === "data: [DONE]") continue;
        
        if (trimmedLine.startsWith("data: ")) {
          try {
            const jsonStr = trimmedLine.slice(6);
            const data = JSON.parse(jsonStr);
            
            // Check for error in stream
            if (data.error) {
              throw new Error(data.error);
            }
            
            const content = data.choices?.[0]?.delta?.content;
            if (content) {
              accumulatedContent += content;
              pendingContentUpdate += content;
              scheduleUpdate();
            }
            
            // Handle thinking/reasoning content (DeepSeek, Ollama, etc.)
            const reasoningContent = 
              data.choices?.[0]?.delta?.reasoning_content ||  // DeepSeek
              data.choices?.[0]?.delta?.reasoning;            // Ollama (qwen3, etc.)
            if (reasoningContent && onThinking) {
              accumulatedThinking += reasoningContent;
              pendingThinkingUpdate += reasoningContent;
              scheduleUpdate();
            }
          } catch (e) {
            // Skip invalid JSON chunks
            if (e instanceof SyntaxError) continue;
            throw e;
          }
        }
      }
    }
    
    // Process any remaining buffer
    if (buffer.trim() && buffer.trim().startsWith("data: ") && buffer.trim() !== "data: [DONE]") {
      try {
        const data = JSON.parse(buffer.trim().slice(6));
        const content = data.choices?.[0]?.delta?.content;
        if (content) {
          accumulatedContent += content;
          pendingContentUpdate += content;
        }
      } catch {
        // Ignore final incomplete chunk
      }
    }
    
    // Final flush - ensure all pending updates are sent
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
    }
    if (pendingContentUpdate) {
      onChunk(pendingContentUpdate, accumulatedContent);
    }
    if (pendingThinkingUpdate && onThinking) {
      onThinking(pendingThinkingUpdate, accumulatedThinking);
    }
  } finally {
    reader.releaseLock();
  }

  onComplete?.(dycpStats);
}

export async function createConversation(userId: string, title?: string, enhancedMode?: boolean): Promise<Conversation> {
  return apiFetch("/api/conversations", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      title: title || "New Conversation",
      enhanced_mode: enhancedMode ?? true,  // Default to enhanced mode
    }),
  });
}

export async function updateConversation(conversationId: number, updates: { title?: string; enhanced_mode?: boolean }): Promise<Conversation> {
  return apiFetch(`/api/conversations/${conversationId}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export async function getConversations(userId: string): Promise<Conversation[]> {
  return apiFetch(`/api/conversations/user/${userId}`);
}

export async function getConversation(conversationId: number): Promise<Conversation> {
  return apiFetch(`/api/conversations/${conversationId}`);
}

export async function getConversationTurns(conversationId: number): Promise<Turn[]> {
  return apiFetch(`/api/conversations/${conversationId}/turns`);
}

export async function deleteConversation(conversationId: number): Promise<void> {
  return apiFetch(`/api/conversations/${conversationId}`, {
    method: "DELETE",
  });
}

export async function getMemoryStats() {
  return apiFetch("/api/stats");
}
