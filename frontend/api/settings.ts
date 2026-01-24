import { apiFetch } from "./http";

export interface UserSettings {
  user_id: string;
  has_api_key: boolean;
  openai_base_url: string | null;
  llm_model: string | null;
  llm_temperature: number | null;
  llm_max_tokens: number | null;
  embedding_model: string | null;
  embedding_dim: number | null;
  // Environment defaults
  default_openai_base_url: string;
  default_llm_model: string;
  default_llm_temperature: number;
  default_llm_max_tokens: number;
  default_embedding_model: string;
  default_embedding_dim: number;
  created_at: string;
  updated_at: string;
}

export interface ValidateApiKeyResponse {
  valid: boolean;
  error: string | null;
}

export interface SaveSettingsParams {
  userId: string;
  openaiApiKey?: string;
  openaiBaseUrl?: string;
  llmModel?: string;
  llmTemperature?: number | null;
  llmMaxTokens?: number | null;
  embeddingModel?: string;
  embeddingDim?: number | null;
}

export async function saveUserSettings(params: SaveSettingsParams): Promise<UserSettings> {
  const body: Record<string, string | number | null | undefined> = {
    user_id: params.userId,
  };
  
  if (params.openaiApiKey !== undefined) {
    body.openai_api_key = params.openaiApiKey;
  }
  if (params.openaiBaseUrl !== undefined) {
    body.openai_base_url = params.openaiBaseUrl;
  }
  if (params.llmModel !== undefined) {
    body.llm_model = params.llmModel;
  }
  if (params.llmTemperature !== undefined) {
    body.llm_temperature = params.llmTemperature;
  }
  if (params.llmMaxTokens !== undefined) {
    body.llm_max_tokens = params.llmMaxTokens;
  }
  if (params.embeddingModel !== undefined) {
    body.embedding_model = params.embeddingModel;
  }
  if (params.embeddingDim !== undefined) {
    body.embedding_dim = params.embeddingDim;
  }
  
  return apiFetch("/api/settings", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getUserSettings(userId: string): Promise<UserSettings> {
  return apiFetch(`/api/settings/${userId}`);
}

export async function validateApiKey(apiKey: string, baseUrl?: string, model?: string): Promise<ValidateApiKeyResponse> {
  const body: Record<string, string> = { api_key: apiKey };
  if (baseUrl) {
    body.base_url = baseUrl;
  }
  if (model) {
    body.model = model;
  }
  return apiFetch("/api/settings/validate-api-key", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function deleteApiKey(userId: string): Promise<void> {
  return apiFetch(`/api/settings/${userId}/api-key`, {
    method: "DELETE",
  });
}

export interface AvailableModels {
  llm_models: string[];
  embedding_models: string[];
}

export async function getAvailableModels(userId: string): Promise<AvailableModels> {
  return apiFetch(`/api/settings/${userId}/models`);
}
