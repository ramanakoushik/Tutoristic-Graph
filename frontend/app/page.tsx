"use client";

import { useEffect, useState } from "react";
import Chat from "@/components/Chat";
import GraphView from "@/components/GraphView";
import Sidebar from "@/components/Sidebar";
import { api, AppState } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { RotateCcw, RefreshCw } from "lucide-react";

function getUserId(): string {
  if (typeof window === "undefined") return "anon";
  let uid = localStorage.getItem("tutoristic_user_id");
  if (!uid) {
    uid = `user_${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}`;
    localStorage.setItem("tutoristic_user_id", uid);
  }
  return uid;
}

export default function Home() {
  const [userId, setUserId] = useState("");
  const [state, setState] = useState<AppState | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"chat" | "graph">("chat");
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    setUserId(getUserId());
  }, []);

  const fetchState = async () => {
    if (!userId) return;
    try {
      const s = await api.getState(userId);
      setState(s);
    } catch (e) {
      console.error("Failed to fetch state", e);
    }
  };

  useEffect(() => {
    if (userId) fetchState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  const sendMessage = async (message: string, goal?: string) => {
    if (!message.trim() || !userId) return;
    setLoading(true);
    try {
      const res = await api.chat({
        user_id: userId,
        message,
        goal: goal || state?.active_page?.goal,
        session_id: sessionId || undefined,
      });
      setState(res.state);
    } catch (e) {
      console.error("Chat error", e);
    } finally {
      setLoading(false);
    }
  };

  const onNewGoal = async (goal: string) => {
    if (!goal.trim() || !userId) return;
    setLoading(true);
    try {
      await api.createTopic({ user_id: userId, goal });
      await fetchState();
    } catch (e) {
      console.error("Create topic error", e);
    } finally {
      setLoading(false);
    }
  };

  const onActivateGoal = async (pageId: string) => {
    if (!userId) return;
    try {
      await api.activatePage({ user_id: userId, page_id: pageId });
      await fetchState();
    } catch (e) {
      console.error("Activate page error", e);
    }
  };

  const onUpdateTodo = async (conceptId: string, status: "locked" | "learning" | "mastered") => {
    if (!userId) return;
    try {
      await api.updateTodo({ user_id: userId, concept_id: conceptId, status });
      await fetchState();
    } catch (e) {
      console.error("Update todo error", e);
    }
  };

  const onReset = async () => {
    if (!userId) return;
    if (!confirm("Reset everything for this user? This wipes graph, pages, todo and chat history."))
      return;
    try {
      await api.reset(userId);
      setSessionId(null);
      await fetchState();
    } catch (e) {
      console.error("Reset error", e);
    }
  };



  const onAnalyze = async () => {
    if (!userId) return;
    setLoading(true);
    try {
      await api.analyzeLinks(userId);
      await fetchState();
    } catch (e) {
      console.error("Analyze error", e);
    } finally {
      setLoading(false);
    }
  };

  const onSavePositions = async (positions: { id: string; x: number; y: number }[]) => {
    if (!userId) return;
    try {
      await api.savePositions({ user_id: userId, positions });
    } catch (e) {
      console.error("Save positions error", e);
    }
  };

  const onAddNode = async (labels: string[], properties: Record<string, any>) => {
    if (!userId) return;
    try {
      await api.addNode({ user_id: userId, labels, properties });
      await fetchState();
    } catch (e) {
      console.error("Add node error", e);
    }
  };

  const onDeleteNode = async (nodeId: string) => {
    if (!userId) return;
    try {
      await api.deleteNode({ user_id: userId, node_id: nodeId });
      await fetchState();
    } catch (e) {
      console.error("Delete node error", e);
    }
  };

  const onAddEdge = async (sourceId: string, targetId: string, edgeType: string, properties?: Record<string, any>) => {
    if (!userId) return;
    try {
      await api.addEdge({ user_id: userId, source_id: sourceId, target_id: targetId, edge_type: edgeType, properties });
      await fetchState();
    } catch (e) {
      console.error("Add edge error", e);
    }
  };

  const onDeleteEdge = async (sourceId: string, targetId: string, edgeType: string) => {
    if (!userId) return;
    try {
      await api.deleteEdge({ user_id: userId, source_id: sourceId, target_id: targetId, edge_type: edgeType });
      await fetchState();
    } catch (e) {
      console.error("Delete edge error", e);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        userId={userId}
        state={state}
        onNewGoal={onNewGoal}
        onUpdateTodo={onUpdateTodo}
        onActivateGoal={onActivateGoal}
        onReset={onReset}
        onRefresh={fetchState}
        loading={loading}
      />
      <main className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b px-4 py-2">
          <div className="flex items-center gap-2">
            <Button
              variant={activeTab === "chat" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("chat")}
            >
              Chat
            </Button>
            <Button
              variant={activeTab === "graph" ? "default" : "ghost"}
              size="sm"
              onClick={() => setActiveTab("graph")}
            >
              Graph View
            </Button>
            <div className="ml-2 text-xs text-muted-foreground">
              {state?.active_page?.goal ? `Goal: ${state.active_page.goal}` : "No active goal"}
              {state?.active_agent ? ` • Agent: ${state.active_agent}` : ""}
              {state?.current_concept?.name ? ` • Current: ${state.current_concept.name}` : ""}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={onAnalyze} disabled={loading}>
              Auto-link
            </Button>
            <Button variant="outline" size="sm" onClick={fetchState} disabled={loading}>
              <RefreshCw className={`mr-1 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
            <Button variant="ghost" size="sm" onClick={onReset} title="Reset this user">
              <RotateCcw className="h-4 w-4" />
            </Button>

          </div>
        </header>
        <div className="flex min-h-0 flex-1">
          {activeTab === "chat" && (
            <Chat
              messages={state?.chat_history || []}
              onSend={sendMessage}
              loading={loading}
              activeGoal={state?.active_page?.goal}
            />
          )}
          {activeTab === "graph" && (
            <GraphView
              nodes={state?.graph.nodes || []}
              edges={state?.graph.edges || []}
              onSavePositions={onSavePositions}
              onAddNode={onAddNode}
              onDeleteNode={onDeleteNode}
              onAddEdge={onAddEdge}
              onDeleteEdge={onDeleteEdge}
            />
          )}
        </div>
      </main>
    </div>
  );
}
