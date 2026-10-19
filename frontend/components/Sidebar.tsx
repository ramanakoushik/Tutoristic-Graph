"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Plus, RefreshCw, RotateCcw, BookOpen, Loader2, ListTodo } from "lucide-react";
import { AppState } from "@/lib/api";
import { cn } from "@/lib/utils";

interface SidebarProps {
  userId: string;
  state: AppState | null;
  onNewGoal: (goal: string) => void;
  onUpdateTodo: (conceptId: string, status: "locked" | "learning" | "mastered") => void;
  onActivateGoal: (pageId: string) => void;
  onReset: () => void;
  onRefresh: () => void;
  loading: boolean;
}

export default function Sidebar({
  state,
  onNewGoal,
  onUpdateTodo,
  onActivateGoal,
  onReset,
  onRefresh,
  loading,
}: SidebarProps) {
  const [newGoal, setNewGoal] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [filter, setFilter] = useState<string>("all");

  const pages = state?.pages || [];
  const todo = state?.todo || [];

  // keep filter valid when pages disappear (reset)
  const activeFilter = filter === "all" || pages.some((p) => p.id === filter) ? filter : "all";

  // a concept can belong to several goals (shared node) — page_ids is the truth
  const scoped = useMemo(() => {
    if (activeFilter === "all") return todo;
    return todo.filter((t) =>
      t.page_ids ? t.page_ids.includes(activeFilter) : t.page_id === activeFilter
    );
  }, [todo, activeFilter]);

  // shared concepts count once toward the top badge
  const unique = useMemo(() => {
    const m = new Map<string, (typeof todo)[number]>();
    for (const t of scoped) if (!m.has(t.id)) m.set(t.id, t);
    return [...m.values()];
  }, [scoped]);
  const masteredCount = unique.filter((t) => t.status === "mastered").length;
  const uniqueTotal = unique.length;

  const groups = useMemo(() => {
    if (activeFilter !== "all") return null; // flat list
    const byPage: {
      id: string;
      title: string;
      active: boolean;
      items: typeof todo;
    }[] = pages.map((p) => ({
      id: p.id,
      title: p.title || p.goal,
      active: !!p.active,
      items: [],
    }));
    const unassigned: typeof todo = [];
    for (const t of todo) {
      const targets = t.page_ids?.length ? t.page_ids : t.page_id ? [t.page_id] : [];
      if (targets.length === 0) unassigned.push(t);
      for (const pid of targets) {
        const grp = byPage.find((g) => g.id === pid);
        if (grp) grp.items.push(t);
      }
    }
    // order each group by that goal's own learning order
    for (const grp of byPage) {
      grp.items.sort(
        (a, b) => (a.orders?.[grp.id] ?? a.order ?? 0) - (b.orders?.[grp.id] ?? b.order ?? 0)
      );
    }
    const out = byPage.filter((g) => g.items.length > 0);
    if (unassigned.length)
      out.push({ id: "__unassigned", title: "Other concepts", active: false, items: unassigned });
    return out;
  }, [todo, pages, activeFilter]);

  const handleAdd = () => {
    if (!newGoal.trim()) return;
    onNewGoal(newGoal.trim());
    setNewGoal("");
    setShowNew(false);
  };

  const statusVariant = (s: string) => {
    if (s === "mastered") return "success" as const;
    if (s === "learning") return "warning" as const;
    return "secondary" as const;
  };

  const statusIcon = (s: string) => {
    if (s === "mastered") return "●";
    if (s === "learning") return "○";
    return "◌";
  };

  const renderItem = (t: (typeof todo)[number]) => (
    <button
      key={t.id}
      onClick={() => {
        if (t.status === "locked") onUpdateTodo(t.id, "learning");
        else if (t.status === "learning") onUpdateTodo(t.id, "mastered");
        else if (t.status === "mastered") onUpdateTodo(t.id, "locked");
      }}
      className={cn(
        "flex w-full items-start justify-between rounded-md border p-2 text-left text-xs transition-colors hover:bg-accent",
        t.status === "mastered" && "border-green-500/40",
        t.status === "learning" && "border-yellow-500/40"
      )}
      title="Click to cycle: locked → learning → mastered"
    >
      <div className="flex items-start gap-2">
        <span className="mt-0.5 select-none">{statusIcon(t.status)}</span>
        <div>
          <div className="font-medium">{t.name}</div>
          {t.description && (
            <div className="line-clamp-2 text-muted-foreground">{t.description}</div>
          )}
        </div>
      </div>
      <Badge variant={statusVariant(t.status)} className="ml-2 shrink-0">
        {t.status}
      </Badge>
    </button>
  );

  return (
    <aside className="flex h-full w-80 flex-col border-r">
      {/* brand bar */}
      <div className="flex items-center justify-between border-b px-3 py-2">
        <span className="flex items-center gap-2 text-sm font-semibold">
          <BookOpen className="h-4 w-4" />
          Tutoristic Graph
        </span>
        <div className="flex gap-1">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onRefresh} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          </Button>
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onReset} title="Reset">
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* TO-DO — top block, owns the goal-switching + new-goal actions */}
      <div className="flex min-h-0 flex-1 flex-col p-2">
        <div className="flex min-h-0 flex-1 flex-col rounded-lg border bg-card">
          <div className="flex items-center justify-between gap-2 border-b p-3 pb-2">
            <span className="flex items-center gap-2 text-sm font-semibold">
              <ListTodo className="h-4 w-4" />
              To-do
              <Badge variant="secondary" className="text-[10px]">
                {masteredCount}/{uniqueTotal} mastered
              </Badge>
            </span>
            <div className="flex items-center gap-1">
              <select
                value={activeFilter}
                onChange={(e) => {
                  const v = e.target.value;
                  setFilter(v);
                  if (v !== "all") onActivateGoal(v);
                }}
                className="h-7 max-w-[130px] truncate rounded-md border bg-background px-1 text-xs"
                title="Switch active goal"
              >
                <option value="all">All goals</option>
                {pages.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.active ? "● " : ""}
                    {p.title || p.goal}
                  </option>
                ))}
              </select>
              <Button
                variant="outline"
                size="icon"
                className="h-7 w-7"
                onClick={() => setShowNew((v) => !v)}
                title="New learning goal"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {showNew && (
            <div className="flex gap-2 border-b p-3 pb-2">
              <Input
                value={newGoal}
                onChange={(e) => setNewGoal(e.target.value)}
                placeholder="e.g. Learn React Hooks"
                className="h-8 text-xs"
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
                autoFocus
              />
              <Button size="sm" className="h-8" onClick={handleAdd} disabled={!newGoal.trim()}>
                Add
              </Button>
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            {todo.length === 0 ? (
              <div className="mt-4 text-center text-xs text-muted-foreground">
                No concepts yet. Hit <span className="font-medium">+</span> or tell the tutor a
                goal like <span className="font-medium">&quot;I want to learn recursion&quot;</span>.
              </div>
            ) : groups ? (
              <div className="flex flex-col gap-4">
                {groups.map((grp) => (
                  <div key={grp.id} className="flex flex-col gap-2">
                    <div className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      <span className="truncate">
                        {grp.active && <span className="text-primary">● </span>}
                        {grp.title}
                      </span>
                      <span className="ml-1 shrink-0">
                        {grp.items.filter((i) => i.status === "mastered").length}/{grp.items.length}
                      </span>
                    </div>
                    {grp.items.map(renderItem)}
                  </div>
                ))}
              </div>
            ) : scoped.length === 0 ? (
              <div className="mt-4 text-center text-xs text-muted-foreground">
                No concepts for this goal yet — chat with the tutor to plan it.
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {[...scoped]
                  .sort(
                    (a, b) =>
                      (a.orders?.[activeFilter] ?? a.order ?? 0) -
                      (b.orders?.[activeFilter] ?? b.order ?? 0)
                  )
                  .map(renderItem)}
              </div>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
