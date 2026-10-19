export interface ChatMessage {
  id: string;
  role: string;
  content: string;
  timestamp: string;
  session_id?: string;
  goal?: string;
}

export interface GraphNode {
  id: string;
  labels: string[];
  properties: Record<string, any>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, any>;
}

export interface TodoItem {
  id: string;
  name: string;
  status: "locked" | "learning" | "mastered";
  explanation?: string;
  description?: string;
  order?: number;
  page_id?: string | null;
  page_ids?: string[];
  orders?: Record<string, number>;
  mastered_at?: string;
  learning_at?: string;
}

export interface PageItem {
  id: string;
  goal: string;
  title?: string;
  active?: boolean;
  created_at?: string;
}

export interface AppState {
  user_id: string;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  pages: PageItem[];
  active_page: PageItem | null;
  todo: TodoItem[];
  chat_history: ChatMessage[];
  sessions: any[];
  active_agent: string | null;
  current_concept: { id: string; name: string } | null;
}

export interface ChatReq {
  user_id: string;
  message: string;
  goal?: string | null;
  is_reply?: boolean | null;
  session_id?: string | null;
}

export interface ChatRes {
  response: string;
  state: AppState;
  next_agent?: string | null;
  score?: number | null;
  feedback?: string | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const api = {
  chat: (body: ChatReq) =>
    fetchJson<ChatRes>("/chat", { method: "POST", body: JSON.stringify(body) }),
  getState: (userId: string) => fetchJson<AppState>(`/state/${userId}`),
  updateTodo: (body: { user_id: string; concept_id: string; status: "locked" | "learning" | "mastered" }) =>
    fetchJson<{ ok: boolean }>("/todo", { method: "PUT", body: JSON.stringify(body) }),
  createTopic: (body: { user_id: string; goal: string; title?: string }) =>
    fetchJson<{ ok: boolean; page_id: string }>("/topic", { method: "POST", body: JSON.stringify(body) }),
  activatePage: (body: { user_id: string; page_id: string }) =>
    fetchJson<{ ok: boolean }>("/page/activate", { method: "PUT", body: JSON.stringify(body) }),
  addNode: (body: { user_id: string; labels: string[]; properties: Record<string, any> }) =>
    fetchJson<{ ok: boolean; node_id: string }>("/graph/node", { method: "POST", body: JSON.stringify(body) }),
  deleteNode: (body: { user_id: string; node_id: string }) =>
    fetchJson<{ ok: boolean }>("/graph/node", { method: "DELETE", body: JSON.stringify(body) }),
  addEdge: (body: { user_id: string; source_id: string; target_id: string; edge_type: string; properties?: Record<string, any> }) =>
    fetchJson<{ ok: boolean }>("/graph/edge", { method: "POST", body: JSON.stringify(body) }),
  deleteEdge: (body: { user_id: string; source_id: string; target_id: string; edge_type: string }) =>
    fetchJson<{ ok: boolean }>("/graph/edge", { method: "DELETE", body: JSON.stringify(body) }),
  savePositions: (body: { user_id: string; positions: { id: string; x: number; y: number }[] }) =>
    fetchJson<{ ok: boolean }>("/graph/positions", { method: "PUT", body: JSON.stringify(body) }),
  analyzeLinks: (userId: string) =>
    fetchJson<{ ok: boolean; created: any[] }>(`/graph/analyze?user_id=${encodeURIComponent(userId)}`, { method: "POST" }),
  reset: (userId: string) => fetchJson<{ ok: boolean }>(`/reset/${userId}`, { method: "POST" }),
};
