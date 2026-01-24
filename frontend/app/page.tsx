"use client";
import { useState, useEffect } from "react";
import LibreChatInterface from "@/components/mainpage";
import SideNavbar from "@/components/sidenavbar";

export default function Home() {
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState<number | undefined>(undefined);
  const [userId, setUserId] = useState<string>("");

  useEffect(() => {
    // Get or create user ID on mount
    let uid = localStorage.getItem('user_id');
    if (!uid) {
      uid = `user_${Date.now()}_${Math.random().toString(36).substring(7)}`;
      localStorage.setItem('user_id', uid);
    }
    setUserId(uid);
  }, []);

  const handleConversationSelect = (conversationId: number) => {
    setCurrentConversationId(conversationId);
    setIsLeftSidebarOpen(false); // Close sidebar on mobile
  };

  const handleNewChat = () => {
    setCurrentConversationId(undefined);
  };

  const handleConversationCreated = (id: number) => {
    setCurrentConversationId(id);
  };

  return (
    <div className="flex h-screen w-full bg-[#0a0a0a] overflow-hidden">
      
      <SideNavbar 
        isOpen={isLeftSidebarOpen} 
        onClose={() => setIsLeftSidebarOpen(false)}
        currentConversationId={currentConversationId}
        onConversationSelect={handleConversationSelect}
        onNewChat={handleNewChat}
        userId={userId}
      />
      <LibreChatInterface 
        onToggleSidebar={() => setIsLeftSidebarOpen(true)}
        conversationId={currentConversationId}
        onConversationCreated={handleConversationCreated}
      />
      
    </div>
  );
}