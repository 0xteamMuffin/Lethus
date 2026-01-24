"use client";
import { useState, useEffect } from 'react';
import { 
  MessageCircle, 
  Plus,
  Trash2,
  X 
} from 'lucide-react';
import { getConversations, createConversation, deleteConversation, type Conversation } from '@/api/message';
import { toast } from 'sonner';

interface SideNavbarProps {
  isOpen?: boolean;
  onClose?: () => void;
  currentConversationId?: number;
  onConversationSelect: (conversationId: number) => void;
  onNewChat: () => void;
  userId: string;
}

const SideNavbar = ({ 
  isOpen = true, 
  onClose,
  currentConversationId,
  onConversationSelect,
  onNewChat,
  userId 
}: SideNavbarProps) => {
  const [activeId, setActiveId] = useState<number | null>(currentConversationId || null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const sidebarClasses = `
    fixed inset-y-0 left-0 z-50 w-64 bg-[#0a0a0a] border-r border-[#2a2a2a] transform transition-transform duration-300 ease-in-out
    md:relative md:translate-x-0 flex flex-col h-full
    ${isOpen ? 'translate-x-0' : '-translate-x-full'}
  `;

  useEffect(() => {
    setActiveId(currentConversationId || null);
  }, [currentConversationId]);

  useEffect(() => {
    if (userId) {
      loadConversations();
    }
  }, [userId]);

  const loadConversations = async () => {
    if (!userId) return;
    
    try {
      setIsLoading(true);
      const convs = await getConversations(userId);
      setConversations(convs);
    } catch (error) {
      console.error("Failed to load conversations:", error);
      toast.error("Failed to load chat history");
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = async () => {
    try {
      const newConv = await createConversation(userId);
      setConversations([newConv, ...conversations]);
      setActiveId(newConv.id);
      onNewChat();
      toast.success("New chat created");
    } catch (error) {
      console.error("Failed to create conversation:", error);
      toast.error("Failed to create new chat");
    }
  };

  const handleSelectConversation = (conversationId: number) => {
    setActiveId(conversationId);
    onConversationSelect(conversationId);
  };

  const handleDeleteConversation = async (conversationId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!confirm("Delete this conversation?")) return;

    try {
      await deleteConversation(conversationId);
      setConversations(conversations.filter(c => c.id !== conversationId));
      
      if (conversationId === activeId) {
        setActiveId(null);
        onNewChat();
      }
      
      toast.success("Chat deleted");
    } catch (error) {
      console.error("Failed to delete conversation:", error);
      toast.error("Failed to delete chat");
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <>
      {isOpen && (
        <div 
          onClick={onClose}
          className="fixed inset-0 bg-black/50 z-40 md:hidden backdrop-blur-sm"
        />
      )}

    <div className={sidebarClasses}>
      
      <div className="flex flex-col gap-3 p-3 pb-2">
        <div className="flex items-center justify-between gap-2">
            
          <button 
            onClick={handleNewChat}
            className="group flex flex-1 cursor-pointer items-center gap-3 rounded-xl border border-[#2a2a2a] bg-[#121212] px-3 py-2.5 transition-all duration-200 hover:bg-[#1a1a1a] hover:border-[#333333]"
          >
            <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-white text-black shadow-sm transition-transform duration-200">
               <Plus size={14} strokeWidth={2.5} />
            </div>
            <span className="text-sm font-medium text-gray-200 group-hover:text-white transition-colors">New chat</span>
          </button>

          <button 
            onClick={onClose}
            className="flex md:hidden h-10 w-10 cursor-pointer items-center justify-center rounded-xl border border-transparent text-gray-400 hover:bg-[#1a1a1a] hover:text-white transition-all duration-200">
            <X size={18} />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-gray-500">Loading...</div>
          </div>
        ) : conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
            <MessageCircle size={32} className="text-gray-700 mb-2" />
            <p className="text-sm text-gray-500">No conversations yet</p>
            <p className="text-xs text-gray-600 mt-1">Start a new chat to begin</p>
          </div>
        ) : (
          <div>
            <div className="px-3 pb-2 text-[10px] font-semibold text-gray-600 uppercase tracking-wider">
                Recent Chats
            </div>
            <div className="space-y-0.5">
              {conversations.map((conv) => (
                <div 
                  key={conv.id}
                  className={`group relative w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-200 cursor-pointer ${
                    activeId === conv.id 
                      ? "bg-[#1a1a1a] text-white border border-[#2a2a2a]" 
                      : "text-gray-400 hover:bg-[#121212] hover:text-gray-200"
                  }`}
                  onClick={() => handleSelectConversation(conv.id)}
                >
                  <MessageCircle size={16} className="shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="truncate font-normal">
                      {conv.title || `Chat ${conv.id}`}
                    </div>
                    <div className="text-[10px] text-gray-600 mt-0.5">
                      {formatDate(conv.updated_at)}
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDeleteConversation(conv.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-[#2a2a2a] transition-all"
                  >
                    <Trash2 size={14} className="text-gray-600 hover:text-red-400" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

    </div>
    </>
  );
};

export default SideNavbar;