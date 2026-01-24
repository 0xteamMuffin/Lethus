"use client";
import React, { useState } from "react";
import { 
  Radar, 
  Clock, 
  Shield, 
  Anchor, 
  Sparkles,
  Layers,
  X 
} from "lucide-react";

interface Memory {
    id: string;
    type: "pinned" | "dynamic" | "recent" | "important";
    content: string;
    relevance?: number;
    timestamp?: string;
    category?: string;
  }
  
  const DUMMY_MEMORIES: Memory[] = [
    { id: "1", type: "pinned", content: "Never use subscriptions in this app", category: "Constraint", relevance: 0.98 },
    { id: "2", type: "pinned", content: "Privacy-focused, no data sharing", category: "Constraint", relevance: 0.95 },
    { id: "3", type: "pinned", content: "Mobile fitness app with offline support", category: "Project Context", relevance: 0.92 },
    { id: "4", type: "important", content: "User prefers one-time purchase model", category: "Decision", relevance: 0.88, timestamp: "2 hours ago" },
    { id: "5", type: "dynamic", content: "Discussed pricing page design requirements", category: "Context Span", relevance: 0.85, timestamp: "5 mins ago" },
    { id: "6", type: "dynamic", content: "Focus on modern, minimalist UI design", category: "Preference", relevance: 0.72, timestamp: "1 hour ago" },
    { id: "7", type: "recent", content: "Asked about context engineering pipeline", category: "Recent Query", timestamp: "Just now" },
    { id: "8", type: "dynamic", content: "Building DYCP-style memory system", category: "Project Goal", relevance: 0.81, timestamp: "3 hours ago" },
    { id: "9", type: "important", content: "Keep color scheme monochrome throughout", category: "Design Rule", relevance: 0.79, timestamp: "15 mins ago" },
    { id: "10", type: "dynamic", content: "Implementing span selection algorithm", category: "Technical Detail", relevance: 0.76, timestamp: "30 mins ago" },
    { id: "11", type: "recent", content: "Updated sidebar with new icons", category: "Recent Action", timestamp: "5 mins ago" },
    { id: "12", type: "dynamic", content: "Token-aware pruning for efficiency", category: "Optimization", relevance: 0.68, timestamp: "2 hours ago" },
    { id: "13", type: "important", content: "Relevance threshold set to 0.7", category: "Configuration", relevance: 0.84, timestamp: "1 hour ago" },
    { id: "14", type: "dynamic", content: "Query embedding using sentence transformers", category: "Technical Stack", relevance: 0.71, timestamp: "4 hours ago" },
    { id: "15", type: "recent", content: "Fixed textarea border issue", category: "Bug Fix", timestamp: "2 mins ago" },
  ];

interface MemorySidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

const MemorySidebar: React.FC<MemorySidebarProps> = ({ isOpen = true, onClose }) => {
  const [selectedType, setSelectedType] = useState<string>("all");
  
  const getMemoryIcon = (type: Memory["type"]) => {
      switch (type) {
        case "pinned": return <Shield size={14} className="text-gray-400" />;
        case "important": return <Anchor size={14} className="text-gray-400" />;
        case "dynamic": return <Layers size={14} className="text-gray-400" />;
        case "recent": return <Clock size={14} className="text-gray-400" />;
        default: return <Radar size={14} className="text-gray-400" />;
      }
    };
  
    const getRelevanceColor = (relevance?: number) => {
      if (!relevance) return "bg-gray-700";
      if (relevance >= 0.9) return "bg-gray-300";
      if (relevance >= 0.75) return "bg-gray-400";
      if (relevance >= 0.6) return "bg-gray-500";
      return "bg-gray-600";
    };

  const filteredMemories = selectedType === "all" 
    ? DUMMY_MEMORIES 
    : DUMMY_MEMORIES.filter(m => m.type === selectedType);

  
  const sidebarClasses = `
    fixed inset-y-0 right-0 z-50 w-80 bg-[#0a0a0a] border-l border-[#2a2a2a] flex flex-col font-sans transform transition-transform duration-300 ease-in-out
    lg:relative lg:translate-x-0 h-full
    ${isOpen ? 'translate-x-0' : 'translate-x-full'}
  `;

  return (
    <>
       {isOpen && (
        <div 
          onClick={onClose}
          className="fixed inset-0 bg-black/50 z-40 lg:hidden backdrop-blur-sm"
        />
      )}

    <div className={sidebarClasses}>
      <div className="px-4 py-4 border-b border-[#2a2a2a]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-[#2a2a2a] border border-[#333333]">
              <Radar size={18} className="text-gray-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">Memory Context</h2>
              <p className="text-[10px] text-gray-500 font-mono">Active Working Memory</p>
            </div>
          </div>
          
           <button 
              onClick={onClose}
              className="flex lg:hidden h-8 w-8 cursor-pointer items-center justify-center rounded-lg hover:bg-[#1a1a1a] text-gray-400"
            >
              <X size={18} />
            </button>
        </div>

        <div className="flex gap-1.5 overflow-x-auto pb-1 scrollbar-none">
          {[
            { label: "All", value: "all" },
            { label: "Pinned", value: "pinned" },
            { label: "Dynamic", value: "dynamic" },
            { label: "Important", value: "important" },
          ].map((tab) => (
            <button
              key={tab.value}
              onClick={() => setSelectedType(tab.value)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all whitespace-nowrap ${
                selectedType === tab.value
                  ? "bg-white text-black"
                  : "bg-[#1a1a1a] text-gray-400 hover:bg-[#212121] hover:text-gray-300"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="px-4 py-3 bg-[#121212] border-b border-[#2a2a2a]">
        <div className="grid grid-cols-3 gap-2">
          <div className="flex flex-col items-center py-2 px-1 bg-[#1a1a1a] rounded-lg">
            <div className="text-lg font-bold text-white">
              {DUMMY_MEMORIES.filter(m => m.type === "pinned").length}
            </div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wide">Pinned</div>
          </div>
          <div className="flex flex-col items-center py-2 px-1 bg-[#1a1a1a] rounded-lg">
            <div className="text-lg font-bold text-white">
              {DUMMY_MEMORIES.filter(m => m.relevance && m.relevance > 0.7).length}
            </div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wide">Relevant</div>
          </div>
          <div className="flex flex-col items-center py-2 px-1 bg-[#1a1a1a] rounded-lg">
            <div className="text-lg font-bold text-white">{DUMMY_MEMORIES.length}</div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wide">Total</div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3">
        <div className="space-y-2">
          {filteredMemories.map((memory) => (
            <div
              key={memory.id}
              className="group relative p-3 bg-[#121212] hover:bg-[#1a1a1a] border border-[#2a2a2a] hover:border-[#333333] rounded-xl transition-all cursor-pointer animate-fade-in"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getMemoryIcon(memory.type)}
                  <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide">
                    {memory.category || memory.type}
                  </span>
                </div>
                {memory.relevance && (
                  <div className="flex items-center gap-1.5">
                    <div className="h-1 w-12 bg-[#2a2a2a] rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getRelevanceColor(memory.relevance)} transition-all`}
                        style={{ width: `${memory.relevance * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              <p className="text-sm text-gray-200 leading-relaxed mb-2">
                {memory.content}
              </p>

              {memory.timestamp && (
                <div className="flex items-center gap-1 text-[10px] text-gray-600">
                  <Clock size={10} />
                  <span>{memory.timestamp}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="px-4 py-3 border-t border-[#2a2a2a] bg-[#0a0a0a]">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Sparkles size={12} className="text-gray-500" />
          <span>Context spans updated dynamically</span>
        </div>
      </div>
    </div>
    </>
  );
};

export default MemorySidebar;