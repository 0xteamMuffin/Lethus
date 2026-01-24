"use client";
import React, { useState, useEffect } from "react";
import { X, Key, Check, AlertCircle, Loader2, Trash2 } from "lucide-react";
import { 
  saveUserSettings, 
  validateApiKey, 
  deleteApiKey,
  getUserSettings,
  getAvailableModels,
  type UserSettings,
  type AvailableModels
} from "@/api/settings";
import { toast } from "sonner";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (hasApiKey: boolean) => void;
  userId: string;
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  onSave,
  userId,
}) => {
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  
  // Settings state
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [llmModel, setLlmModel] = useState<string>("");
  const [llmTemperature, setLlmTemperature] = useState<string>("");
  const [llmMaxTokens, setLlmMaxTokens] = useState<string>("");
  const [embeddingModel, setEmbeddingModel] = useState<string>("");
  const [embeddingDim, setEmbeddingDim] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<AvailableModels | null>(null);
  
  // Validation state
  const [apiKeyValid, setApiKeyValid] = useState<boolean | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Load settings when modal opens
  useEffect(() => {
    if (isOpen && userId) {
      loadSettings();
    }
  }, [isOpen, userId]);

  const loadSettings = async () => {
    try {
      const userSettings = await getUserSettings(userId);
      setSettings(userSettings);
      setBaseUrl(userSettings.openai_base_url || "");
      setLlmModel(userSettings.llm_model || "");
      setLlmTemperature(userSettings.llm_temperature?.toString() || "");
      setLlmMaxTokens(userSettings.llm_max_tokens?.toString() || "");
      setEmbeddingModel(userSettings.embedding_model || "");
      setEmbeddingDim(userSettings.embedding_dim?.toString() || "");
      
      // Try to fetch available models if user has API key
      if (userSettings.has_api_key) {
        loadAvailableModels();
      }
    } catch (error) {
      console.error("Failed to load settings:", error);
    }
  };

  const loadAvailableModels = async () => {
    try {
      const models = await getAvailableModels(userId);
      setAvailableModels(models);
    } catch (error) {
      console.error("Failed to fetch available models:", error);
      // Silently fail - user can still type custom models
    }
  };

  const handleValidateApiKey = async () => {
    if (!apiKey.trim()) return;
    
    setIsValidating(true);
    setApiKeyValid(null);
    setValidationError(null);
    
    try {
      // Pass base_url and model if user entered them, otherwise use saved settings or let backend use defaults
      const effectiveBaseUrl = baseUrl.trim() || settings?.openai_base_url || undefined;
      const effectiveModel = llmModel.trim() || settings?.llm_model || undefined;
      const result = await validateApiKey(apiKey.trim(), effectiveBaseUrl, effectiveModel);
      setApiKeyValid(result.valid);
      if (!result.valid) {
        setValidationError(result.error || "Invalid API key");
      }
    } catch (error) {
      setApiKeyValid(false);
      setValidationError("Failed to validate API key");
    } finally {
      setIsValidating(false);
    }
  };

  const handleSaveApiKey = async () => {
    if (!apiKey.trim()) return;
    
    // Validate first if not yet validated
    if (apiKeyValid === null) {
      await handleValidateApiKey();
      if (!apiKeyValid) return;
    }
    
    setIsSaving(true);
    try {
      const result = await saveUserSettings({
        userId: userId,
        openaiApiKey: apiKey.trim(),
        openaiBaseUrl: baseUrl.trim() || "",
      });
      
      setSettings(result);
      setApiKey("");
      setBaseUrl(result.openai_base_url || "");
      setApiKeyValid(null);
      toast.success("API settings saved successfully!");
      onSave(true);
    } catch (error) {
      console.error("Failed to save API settings:", error);
      toast.error("Failed to save API settings");
    } finally {
      setIsSaving(false);
    }
  };

  const handleSaveModels = async () => {
    setIsSaving(true);
    try {
      const result = await saveUserSettings({
        userId: userId,
        openaiBaseUrl: baseUrl || "",
        llmModel: llmModel || "",
        llmTemperature: llmTemperature ? parseFloat(llmTemperature) : null,
        llmMaxTokens: llmMaxTokens ? parseInt(llmMaxTokens, 10) : null,
        embeddingModel: embeddingModel || "",
        embeddingDim: embeddingDim ? parseInt(embeddingDim, 10) : null,
      });
      
      setSettings(result);
      toast.success("Preferences saved!");
    } catch (error) {
      console.error("Failed to save preferences:", error);
      toast.error("Failed to save preferences");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteApiKey = async () => {
    if (!confirm("Are you sure you want to remove your API key?")) return;
    
    setIsDeleting(true);
    try {
      await deleteApiKey(userId);
      setSettings(prev => prev ? { ...prev, has_api_key: false } : null);
      toast.success("API key removed");
      onSave(false);
    } catch (error) {
      console.error("Failed to delete API key:", error);
      toast.error("Failed to remove API key");
    } finally {
      setIsDeleting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 animate-fade-in"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div
          className="bg-[#0a0a0a] border border-[#2a2a2a] rounded-2xl w-full max-w-lg shadow-2xl animate-fade-in max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a2a] sticky top-0 bg-[#0a0a0a]">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-[#1a1a1a] border border-[#2a2a2a]">
                <Key size={20} className="text-gray-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Settings</h2>
                <p className="text-xs text-gray-500">API key and model preferences</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-[#1a1a1a] text-gray-400 hover:text-white transition-all"
            >
              <X size={20} />
            </button>
          </div>

          {/* Content */}
          <div className="px-6 py-6 space-y-6">
            {/* API Configuration Section */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-white">OpenAI API Configuration</h3>
              
              {settings?.has_api_key ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between px-4 py-3 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl">
                    <div className="flex items-center gap-2">
                      <Check size={16} className="text-green-500" />
                      <span className="text-sm text-gray-300">API key configured</span>
                    </div>
                    <button
                      onClick={handleDeleteApiKey}
                      disabled={isDeleting}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-all"
                    >
                      {isDeleting ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                      Remove
                    </button>
                  </div>
                  
                  {/* Update API Key */}
                  <div className="pt-2">
                    <label className="block text-xs text-gray-500 mb-2">
                      Update API Key (optional)
                    </label>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <input
                          type={showKey ? "text" : "password"}
                          value={apiKey}
                          onChange={(e) => {
                            setApiKey(e.target.value);
                            setApiKeyValid(null);
                            setValidationError(null);
                          }}
                          placeholder="sk-..."
                          className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                        />
                        <button
                          type="button"
                          onClick={() => setShowKey(!showKey)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                        >
                          {showKey ? "Hide" : "Show"}
                        </button>
                      </div>
                      <button
                        onClick={handleSaveApiKey}
                        disabled={!apiKey.trim() || isSaving}
                        className="px-4 py-2 rounded-xl text-sm font-medium bg-white text-black hover:bg-gray-200 disabled:bg-[#2a2a2a] disabled:text-gray-600 disabled:cursor-not-allowed transition-all"
                      >
                        {isSaving ? <Loader2 size={16} className="animate-spin" /> : "Update"}
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {/* Base URL (shown first when no API key) */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-2">
                      API Base URL (optional)
                    </label>
                    <input
                      type="text"
                      value={baseUrl}
                      onChange={(e) => setBaseUrl(e.target.value)}
                      placeholder={settings?.default_openai_base_url || "https://api.openai.com/v1"}
                      className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                    />
                    <p className="mt-1 text-xs text-gray-600">
                      For OpenAI-compatible APIs (e.g., GitHub Models, Azure)
                    </p>
                  </div>
                  
                  {/* API Key */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-2">
                      API Key
                    </label>
                    <div className="relative">
                      <input
                        type={showKey ? "text" : "password"}
                        value={apiKey}
                        onChange={(e) => {
                          setApiKey(e.target.value);
                          setApiKeyValid(null);
                          setValidationError(null);
                        }}
                        placeholder="sk-..."
                        className={`w-full bg-[#121212] border rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:outline-none transition-all ${
                          apiKeyValid === true ? "border-green-500" : 
                          apiKeyValid === false ? "border-red-500" : 
                          "border-[#2a2a2a] focus:border-[#404040]"
                        }`}
                      />
                      <button
                        type="button"
                        onClick={() => setShowKey(!showKey)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                      >
                        {showKey ? "Hide" : "Show"}
                      </button>
                    </div>
                  </div>
                  
                  {validationError && (
                    <p className="text-xs text-red-400 flex items-center gap-1">
                      <AlertCircle size={12} />
                      {validationError}
                    </p>
                  )}
                  
                  {apiKeyValid === true && (
                    <p className="text-xs text-green-400 flex items-center gap-1">
                      <Check size={12} />
                      API key is valid
                    </p>
                  )}
                  
                  <div className="flex gap-2">
                    <button
                      onClick={handleValidateApiKey}
                      disabled={!apiKey.trim() || isValidating}
                      className="flex-1 px-4 py-2 rounded-xl text-sm font-medium border border-[#2a2a2a] text-gray-300 hover:bg-[#1a1a1a] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {isValidating ? (
                        <span className="flex items-center justify-center gap-2">
                          <Loader2 size={14} className="animate-spin" />
                          Validating...
                        </span>
                      ) : "Validate"}
                    </button>
                    <button
                      onClick={handleSaveApiKey}
                      disabled={!apiKey.trim() || isSaving}
                      className="flex-1 px-4 py-2 rounded-xl text-sm font-medium bg-white text-black hover:bg-gray-200 disabled:bg-[#2a2a2a] disabled:text-gray-600 disabled:cursor-not-allowed transition-all"
                    >
                      {isSaving ? (
                        <span className="flex items-center justify-center gap-2">
                          <Loader2 size={14} className="animate-spin" />
                          Saving...
                        </span>
                      ) : "Save"}
                    </button>
                  </div>
                  
                  <p className="text-xs text-gray-500 flex items-start gap-2">
                    <AlertCircle size={12} className="mt-0.5 shrink-0" />
                    <span>
                      Your API key is securely stored and used for all chat requests.
                    </span>
                  </p>
                </div>
              )}
            </div>

            {/* Model Selection Section */}
            {settings?.has_api_key && (
              <div className="space-y-4 pt-4 border-t border-[#2a2a2a]">
                <h3 className="text-sm font-semibold text-white">Preferences</h3>
                
                <div className="space-y-4">
                  {/* Base URL */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-2">
                      API Base URL
                    </label>
                    <input
                      type="text"
                      value={baseUrl}
                      onChange={(e) => setBaseUrl(e.target.value)}
                      placeholder={settings?.default_openai_base_url || "https://api.openai.com/v1"}
                      className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                    />
                    <p className="mt-1 text-xs text-gray-600">
                      Default: {settings?.default_openai_base_url}
                    </p>
                  </div>
                  
                  {/* LLM Model */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-2">
                      Chat Model
                    </label>
                    <input
                      type="text"
                      list="llm-models-list"
                      value={llmModel}
                      onChange={(e) => setLlmModel(e.target.value)}
                      placeholder={settings?.default_llm_model || "e.g., openai/gpt-4o-mini"}
                      className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                    />
                    <datalist id="llm-models-list">
                      {availableModels?.llm_models.map((model) => (
                        <option key={model} value={model} />
                      ))}
                    </datalist>
                    <p className="mt-1 text-xs text-gray-600">
                      Default: {settings?.default_llm_model}
                    </p>
                  </div>
                  
                  {/* Temperature and Max Tokens row */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs text-gray-400 mb-2">
                        Temperature
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        min="0"
                        max="2"
                        value={llmTemperature}
                        onChange={(e) => setLlmTemperature(e.target.value)}
                        placeholder={settings?.default_llm_temperature?.toString() || "0.7"}
                        className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                      />
                      <p className="mt-1 text-xs text-gray-600">
                        Default: {settings?.default_llm_temperature}
                      </p>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400 mb-2">
                        Max Tokens
                      </label>
                      <input
                        type="number"
                        step="100"
                        min="1"
                        value={llmMaxTokens}
                        onChange={(e) => setLlmMaxTokens(e.target.value)}
                        placeholder={settings?.default_llm_max_tokens?.toString() || "1000"}
                        className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                      />
                      <p className="mt-1 text-xs text-gray-600">
                        Default: {settings?.default_llm_max_tokens}
                      </p>
                    </div>
                  </div>
                  
                  {/* Embedding Model */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-2">
                      Embedding Model
                    </label>
                    <input
                      type="text"
                      list="embedding-models-list"
                      value={embeddingModel}
                      onChange={(e) => setEmbeddingModel(e.target.value)}
                      placeholder={settings?.default_embedding_model || "e.g., text-embedding-3-small"}
                      className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                    />
                    <datalist id="embedding-models-list">
                      {availableModels?.embedding_models.map((model) => (
                        <option key={model} value={model} />
                      ))}
                    </datalist>
                    <p className="mt-1 text-xs text-gray-600">
                      Default: {settings?.default_embedding_model}
                    </p>
                  </div>
                  
                  {/* Embedding Dimension */}
                  <div>
                    <label className="block text-xs text-gray-400 mb-2">
                      Embedding Dimension
                    </label>
                    <input
                      type="number"
                      min="1"
                      value={embeddingDim}
                      onChange={(e) => setEmbeddingDim(e.target.value)}
                      placeholder={settings?.default_embedding_dim?.toString() || "1536"}
                      className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                    />
                    <p className="mt-1 text-xs text-gray-600">
                      Default: {settings?.default_embedding_dim}
                    </p>
                  </div>
                  
                  <button
                    onClick={handleSaveModels}
                    disabled={isSaving}
                    className="w-full px-4 py-2.5 rounded-xl text-sm font-medium bg-[#1a1a1a] border border-[#2a2a2a] text-white hover:bg-[#212121] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    {isSaving ? (
                      <span className="flex items-center justify-center gap-2">
                        <Loader2 size={14} className="animate-spin" />
                        Saving...
                      </span>
                    ) : "Save Preferences"}
                  </button>
                  
                  <p className="text-xs text-gray-500">
                    Leave empty to use the default settings from server configuration.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end px-6 py-4 border-t border-[#2a2a2a] sticky bottom-0 bg-[#0a0a0a]">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-[#1a1a1a] transition-all"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default SettingsModal;
