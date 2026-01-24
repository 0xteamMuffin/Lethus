"use client";
import React, { useState, useEffect } from "react";
import { X, Key, Check, AlertCircle } from "lucide-react";
import { saveUserSettings } from "@/api/settings";
import { toast } from "sonner";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (apiKey: string) => void;
  currentApiKey?: string;
  userId: string;
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  onSave,
  currentApiKey = "",
  userId,
}) => {
  const [apiKey, setApiKey] = useState(currentApiKey);
  const [showKey, setShowKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    setApiKey(currentApiKey);
  }, [currentApiKey]);

  const handleSave = async () => {
    if (!apiKey.trim()) return;
    
    setIsSaving(true);
    try {
      await saveUserSettings({
        userId: userId,
        openaiApiKey: apiKey.trim(),
      });
      
      onSave(apiKey.trim());
      toast.success("API key saved successfully!");
      onClose();
    } catch (error) {
      console.error("Failed to save API key:", error);
      toast.error("Failed to save API key");
    } finally {
      setIsSaving(false);
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
          className="bg-[#0a0a0a] border border-[#2a2a2a] rounded-2xl w-full max-w-md shadow-2xl animate-fade-in"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a2a]">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-[#1a1a1a] border border-[#2a2a2a]">
                <Key size={20} className="text-gray-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Settings</h2>
                <p className="text-xs text-gray-500">Configure your API key</p>
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
          <div className="px-6 py-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                OpenAI API Key
              </label>
              <div className="relative">
                <input
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-[#121212] border border-[#2a2a2a] rounded-xl px-4 py-3 text-white placeholder-gray-600 focus:border-[#404040] focus:outline-none transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                >
                  {showKey ? "Hide" : "Show"}
                </button>
              </div>
              <p className="mt-2 text-xs text-gray-500 flex items-start gap-2">
                <AlertCircle size={12} className="mt-0.5 shrink-0" />
                <span>
                  Your API key is securely stored in the database and used for all your chat requests.
                </span>
              </p>
            </div>

            {currentApiKey && (
              <div className="flex items-center gap-2 px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg">
                <Check size={14} className="text-green-500" />
                <span className="text-xs text-gray-400">
                  API key configured ({currentApiKey.slice(0, 7)}...{currentApiKey.slice(-4)})
                </span>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#2a2a2a]">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-[#1a1a1a] transition-all"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!apiKey.trim() || isSaving}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                apiKey.trim() && !isSaving
                  ? "bg-white text-black hover:bg-gray-200"
                  : "bg-[#2a2a2a] text-gray-600 cursor-not-allowed"
              }`}
            >
              {isSaving ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default SettingsModal;
