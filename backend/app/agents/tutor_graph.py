from __future__ import annotations

import json
from collections import deque
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.memory.graph_memory import GraphMemory


class TutorState(TypedDict, total=False):
    user_id: str
    message: str
    goal: Optional[str]
    is_reply: Optional[bool]
    is_new_goal: bool
    session_id: Optional[str]
    next_agent: Optional[str]
    score: Optional[int]
    feedback: Optional[str]
    answer: Optional[str]
    concept_name: Optional[str]
    concept_id: Optional[str]
    path: List[str]
    notes: Optional[str]
    response: Optional[str]
    retries: int


def _extract_json_list(text: str) -> List[Any]:
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _extract_json_obj(text: str) -> Dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


class TutorGraph:
    """LangGraph state machine: Planner -> Instructor -> Examiner, all sharing one FalkorDB graph."""

    def __init__(self) -> None:
        self.mem = GraphMemory()
        self.llm = None

    def _llm(self):
        if self.llm is None:
            self.llm = ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.model,
                temperature=settings.temperature,
            )
        return self.llm

    # ------------------------------------------------------------- planner
    def planner(self, state: TutorState) -> TutorState:
        uid = state["user_id"]
        msg = state.get("message") or ""
        goal = state.get("goal") or msg
        if not goal:
            goal = "Learn"
        self.mem.set_active_agent(uid, "planner")
        self.mem.record_interaction(uid, None, "planner", "start", notes=f"goal={goal}")

        # Page selection: a brand-new goal gets its own page (or re-activates an
        # existing page with the same goal); otherwise continue the active one.
        pages = self.mem.get_state(uid)["pages"]
        active = next((p for p in pages if p.get("active")), None)
        goal_norm = goal.strip().lower()
        same_goal_page = next(
            (p for p in pages if (p.get("goal") or "").strip().lower() == goal_norm), None
        )
        if same_goal_page:
            page_id = same_goal_page["id"]
            if not same_goal_page.get("active"):
                self.mem.set_active_page(uid, page_id)
        elif state.get("is_new_goal") or not active:
            page_id = self.mem.create_page(uid, goal)
        else:
            page_id = active["id"]

        # Build concept path ordered by REQUIRES edges (topological)
        g = self.mem._graph(uid)
        cres = g.query(
            "MATCH (p:Page {id:$pid})-[:HAS_CONCEPT]->(c:Concept) RETURN c.name, c.id, c.order",
            {"pid": page_id},
        )
        concepts_list = [{"name": r[0], "id": r[1], "order": r[2] or 0} for r in cres.result_set]

        path_names: List[str] = []
        if concepts_list:
            nodes_set = {c["name"] for c in concepts_list}
            indeg: Dict[str, int] = {n: 0 for n in nodes_set}
            adj: Dict[str, List[str]] = {n: [] for n in nodes_set}
            eres = g.query(
                "MATCH (a:Concept)-[r:REQUIRES|PREREQUISITE_OF|DEPENDS_ON]->(b:Concept) "
                "RETURN a.name, b.name",
                {},
            )
            # A REQUIRES B => B is prerequisite => learn B first => edge B->A in topo graph
            for src, dst in eres.result_set:
                if src in nodes_set and dst in nodes_set:
                    # src requires dst => dst before src => adj[dst].append(src)
                    adj.setdefault(dst, []).append(src)
                    indeg[src] = indeg.get(src, 0) + 1
            qd = deque([n for n in nodes_set if indeg.get(n, 0) == 0])
            order_list: List[str] = []
            while qd:
                cur = qd.popleft()
                order_list.append(cur)
                for nb in adj.get(cur, []):
                    indeg[nb] -= 1
                    if indeg[nb] == 0:
                        qd.append(nb)
            for n in nodes_set:
                if n not in order_list:
                    order_list.append(n)
            path_names = order_list
        else:
            # New goal on a fresh page: plan WITH the existing graph in view so
            # overlapping concepts reuse the same node instead of duplicating it.
            allc = g.query(
                "MATCH (c:Concept) RETURN c.name, c.status LIMIT 300",
                {},
            )
            existing = [
                {"name": r[0], "status": r[1] or "locked"}
                for r in allc.result_set
                if r[0]
            ]
            existing_names = {e["name"].strip().lower(): e["name"] for e in existing}
            sys = (
                "You are a curriculum planner maintaining ONE shared knowledge graph "
                "for a learner. Break the goal into 3-6 concepts in learning order "
                "(prerequisites first).\n"
                "You are given the EXISTING NODES in the graph. If an existing node "
                "already covers something you would add, REUSE it: copy its name "
                "EXACTLY as written (case-sensitive) and set exists=true. Only invent "
                "a new name (exists=false) when nothing overlaps. Strongly prefer "
                "reuse over near-duplicate new nodes — the graph must stay minimal.\n"
                'Return ONLY JSON: [{"name": "...", "exists": true|false}, ...]'
            )
            ex_lines = "\n".join(
                f"- {e['name']} ({e['status']})" for e in existing
            ) or "(graph is empty)"
            user_p = f"Goal: {goal}\n\nEXISTING NODES ({len(existing)}):\n{ex_lines}"
            try:
                r = self._llm().invoke(
                    [SystemMessage(content=sys), HumanMessage(content=user_p)]
                )
                raw = _extract_json_list(r.content if hasattr(r, "content") else str(r))
            except Exception:
                raw = []
            path_names = []
            for item in raw[:8]:
                if isinstance(item, str) and item.strip():
                    path_names.append(item.strip())
                elif isinstance(item, dict):
                    nm = str(item.get("name") or "").strip()
                    if not nm:
                        continue
                    if item.get("exists") and nm.strip().lower() in existing_names:
                        # force the canonical spelling of the existing node
                        nm = existing_names[nm.strip().lower()]
                    path_names.append(nm)
            if not path_names:
                path_names = [goal]

        for i, nm in enumerate(path_names):
            # MERGE on (user_id, name): matches an existing node → reused, else new
            self.mem.upsert_concept(
                uid, nm, order=i, status="locked", page_id=page_id, page_order=i
            )

        # Point current concept at this page's target (keep if already on this page)
        st = self.mem.get_state(uid)
        on_active = [t for t in st["todo"] if t.get("page_id") == page_id]
        on_active_sorted = sorted(on_active, key=lambda x: x.get("order") or 0)
        learning_c = next((t for t in on_active_sorted if t["status"] == "learning"), None)
        locked_c = next((t for t in on_active_sorted if t["status"] == "locked"), None)
        cur = st.get("current_concept")
        cur_on_active = bool(cur and any(t["id"] == cur["id"] for t in on_active))
        target = learning_c or locked_c
        if target and not cur_on_active:
            self.mem.set_concept_status(uid, target["id"], "learning")
            self.mem.set_current_concept(uid, target["id"], target["name"])
            self.mem.record_interaction(uid, target["name"], "planner", "set_learning")

        self.mem.record_interaction(uid, None, "planner", "planned_path", notes=json.dumps(path_names))
        self.mem.set_active_agent(uid, "linker")
        return {
            **state,
            "goal": goal,
            "next_agent": "linker",
            "path": path_names,
            "notes": f"Planned {len(path_names)} concepts",
        }

    # ------------------------------------------------------------- linker
    def linker(self, state: TutorState) -> TutorState:
        """Traversal agent: reads every node in the shared graph and uses the
        LLM to wire the newly-planned concepts into it (REQUIRES / RELATED_TO),
        so each new goal connects to what the learner already knows."""
        uid = state["user_id"]
        path = [p for p in (state.get("path") or []) if isinstance(p, str)]
        goal = state.get("goal") or ""
        self.mem.set_active_agent(uid, "linker")

        g = self.mem._graph(uid)
        cres = g.query(
            "MATCH (c:Concept) "
            "RETURN c.id, c.name, LEFT(COALESCE(c.description, ''), 160), c.status "
            "LIMIT 300",
            {},
        )
        concepts = [
            {"id": r[0], "name": r[1], "description": r[2] or "", "status": r[3] or "locked"}
            for r in cres.result_set
            if r[1]
        ]
        if len(concepts) < 2 or not settings.groq_api_key:
            self.mem.set_active_agent(uid, "instructor")
            return {**state, "next_agent": "instructor"}

        # existing knowledge edges, so we don't propose duplicates
        eres = g.query(
            "MATCH (a:Concept)-[r:REQUIRES|RELATED_TO]->(b:Concept) "
            "RETURN a.id, b.id, TYPE(r)",
            {},
        )
        existing = {(r[0], r[1], r[2]) for r in eres.result_set}

        name_to_id = {c["name"].strip().lower(): c["id"] for c in concepts}
        new_set = {p.strip().lower() for p in path}

        sys_prompt = (
            "You are a graph-linking agent. You receive the full set of concept nodes "
            "in a learner's knowledge graph and the ordered path of NEW concepts just "
            "planned for a goal. Decide which edges to create so the graph connects "
            "related ideas across goals.\n"
            "Rules:\n"
            "- (source) -[:REQUIRES]-> (target) means source needs target learned first.\n"
            "- Use RELATED_TO only for loose association (no strict prerequisite).\n"
            "- Chain the new path concepts in learning order where a real dependency exists.\n"
            "- Connect new concepts to existing nodes when the link is genuinely useful "
            "(shared prerequisite, follow-up, overlap). Do NOT force links.\n"
            '- Names must be copied EXACTLY from the provided list.\n'
            "- Return ONLY a JSON array, max 12 edges: "
            '[{"source":"...","target":"...","edge_type":"REQUIRES"}]'
        )
        node_lines = "\n".join(
            f"- {c['name']} ({c['status']})" + (f" — {c['description']}" if c["description"] else "")
            for c in concepts
        )
        order_lines = "\n".join(f"{i+1}. {p}" for i, p in enumerate(path)) or "(none)"
        user_prompt = (
            f"Goal: {goal}\n\n"
            f"New path (learning order, first = prerequisite):\n{order_lines}\n\n"
            f"All {len(concepts)} nodes in the graph:\n{node_lines}"
        )

        created: List[Dict[str, str]] = []
        try:
            r = self._llm().invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            raw = _extract_json_list(r.content if hasattr(r, "content") else str(r))
        except Exception:
            raw = []

        for e in raw[:20]:
            try:
                src_name = str(e.get("source") or "").strip().lower()
                tgt_name = str(e.get("target") or "").strip().lower()
                etype = str(e.get("edge_type") or "REQUIRES").upper()
                if etype not in ("REQUIRES", "RELATED_TO"):
                    etype = "REQUIRES"
                sid, tid = name_to_id.get(src_name), name_to_id.get(tgt_name)
                if not sid or not tid or sid == tid:
                    continue
                if (sid, tid, etype) in existing:
                    continue
                self.mem.add_edge(uid, sid, tid, etype)
                existing.add((sid, tid, etype))
                created.append(
                    {"source": e.get("source", ""), "target": e.get("target", ""), "type": etype}
                )
                if len(created) >= 12:
                    break
            except Exception:
                continue

        # note when a brand-new goal wired into older material
        linked_to_old = [
            c for c in created
            if c["source"].strip().lower() not in new_set or c["target"].strip().lower() not in new_set
        ]
        notes = f"created {len(created)} edges"
        if linked_to_old:
            notes += f" ({len(linked_to_old)} connect to existing nodes)"
        self.mem.record_interaction(uid, None, "linker", "linked", notes=notes)
        self.mem.set_active_agent(uid, "instructor")
        return {
            **state,
            "notes": notes,
            "next_agent": "instructor",
        }

    # ---------------------------------------------------------- instructor
    def instructor(self, state: TutorState) -> TutorState:
        uid = state["user_id"]
        msg = state.get("message") or ""
        self.mem.set_active_agent(uid, "instructor")

        st = self.mem.get_state(uid)
        cur = st.get("current_concept")
        concept_name = cur["name"] if cur else None
        concept_id = cur["id"] if cur else None

        if not concept_name:
            cand = next((t for t in st["todo"] if t["status"] == "learning"), None) or next(
                (t for t in st["todo"] if t["status"] == "locked"), None
            )
            if cand:
                concept_name, concept_id = cand["name"], cand["id"]
                self.mem.set_current_concept(uid, concept_id, concept_name)
                self.mem.set_concept_status(uid, concept_id, "learning")

        if not concept_name:
            resp = (
                "I need a goal first. Tell me what you'd like to learn "
                "(e.g. 'I want to learn recursion')."
            )
            self.mem.add_message(uid, "assistant", resp, session_id=state.get("session_id"), goal=state.get("goal"))
            self.mem.record_interaction(uid, None, "instructor", "no_concept", notes=resp)
            return {**state, "response": resp, "next_agent": END}

        succ = self.mem.get_successful_strategies(uid, concept_name)
        succ_str = (
            "\n".join(f"- {s}" for s in succ)
            if succ
            else "- None yet (use a clear, scaffolded explanation first)"
        )

        hist_msgs = self.mem.get_chat_messages(uid, limit=40)
        sys_inst = (
            "You are an expert tutor (Instructor agent). Teach the CURRENT CONCEPT adaptively, "
            "using prior successful strategies when helpful. Be concise and structured, use worked "
            "examples, call out common pitfalls, and end with a short check question or small task "
            "for the learner. Do NOT just hand over the final answer. Use markdown."
        )
        user_p = (
            f"Current concept: {concept_name}\n\n"
            f"Strategies that worked for this learner before:\n{succ_str}\n\n"
            f"Latest learner message (may be empty):\n{msg}\n\n"
            "Teach this concept now: brief intuition -> definition -> worked example -> "
            "common pitfall -> check question/task."
        )
        try:
            chain_msgs = [SystemMessage(content=sys_inst)] + hist_msgs[-20:] + [HumanMessage(content=user_p)]
            r = self._llm().invoke(chain_msgs)
            instr_text = r.content if hasattr(r, "content") else str(r)
        except Exception:
            instr_text = (
                f"### {concept_name}\n\nLet's learn this step by step.\n\n"
                f"In your own words: what is {concept_name}? I'll ask a short check next."
            )

        strat_attempt = "Scaffolded explanation + worked example + check question"
        self.mem.record_taught_strategy(uid, concept_name, strat_attempt, success=False)
        self.mem.record_interaction(uid, concept_name, "instructor", "taught", notes=strat_attempt)
        self.mem.add_message(uid, "assistant", instr_text, session_id=state.get("session_id"), goal=state.get("goal"))
        self.mem.set_active_agent(uid, "examiner")
        return {
            **state,
            "concept_name": concept_name,
            "concept_id": concept_id,
            "response": instr_text,
            "next_agent": END,
            "answer": msg,
        }

    # ------------------------------------------------------------ examiner
    def examiner(self, state: TutorState) -> TutorState:
        uid = state["user_id"]
        learner_answer = state.get("answer") or state.get("message") or ""
        self.mem.set_active_agent(uid, "examiner")

        st = self.mem.get_state(uid)
        cur = st.get("current_concept")
        concept_name = cur["name"] if cur else state.get("concept_name")
        concept_id = cur["id"] if cur else state.get("concept_id")

        if not concept_name:
            resp = "Nothing to assess yet. What would you like to learn?"
            self.mem.add_message(uid, "assistant", resp, session_id=state.get("session_id"), goal=state.get("goal"))
            return {**state, "response": resp, "next_agent": END}

        # Skip grading if the user just said hi / no substantive answer
        if len(learner_answer.strip()) < 5:
            st2 = self.mem.get_state(uid)
            active_pid = st2.get("active_page", {}).get("id") if st2.get("active_page") else None
            next_c = next(
                (
                    t
                    for t in sorted(st2["todo"], key=lambda x: x.get("order") or 0)
                    if (not active_pid or t.get("page_id") == active_pid) and t["status"] == "locked"
                ),
                None,
            ) or next((t for t in st2["todo"] if t["status"] == "locked"), None)
            if next_c:
                self.mem.set_concept_status(uid, next_c["id"], "learning")
                self.mem.set_current_concept(uid, next_c["id"], next_c["name"])
                resp = f"**Next up:** _{next_c['name']}_.\n\nAnswer the check question when you're ready."
                self.mem.add_message(uid, "assistant", resp, session_id=state.get("session_id"), goal=state.get("goal"))
                return {**state, "response": resp, "next_agent": "instructor"}
            return {**state, "response": "", "next_agent": END}

        sys_ex = (
            "You are an Examiner agent grading a short answer against a concept. Be fair and "
            "constructive. Return ONLY JSON: "
            '{"score": 0-100, "verdict": "pass|needs_review|fail", "feedback": "...", '
            '"correct": true|false, "reason": "..."}'
        )
        ctx = f"Concept: {concept_name}\nLearner answer:\n{learner_answer}\n\nGrade it. 70 or above = pass."
        try:
            r = self._llm().invoke([SystemMessage(content=sys_ex), HumanMessage(content=ctx)])
            jtxt = r.content if hasattr(r, "content") else str(r)
            gj = _extract_json_obj(jtxt)
        except Exception:
            gj = {"score": 50, "verdict": "needs_review", "feedback": "Couldn't parse grading—try a clearer answer.", "correct": False}

        score = int(gj.get("score") or 0)
        feedback = gj.get("feedback") or ""
        passed = score >= 70 or gj.get("verdict") == "pass"

        if passed and concept_id and concept_name:
            self.mem.set_concept_status(uid, concept_id, "mastered")
            self.mem.mark_strategy_success(uid, concept_name)
            self.mem.set_current_concept(uid, None, None)
            self.mem.record_interaction(uid, concept_name, "examiner", "passed", notes=f"score={score}")

            st2 = self.mem.get_state(uid)
            active_pid = st2.get("active_page", {}).get("id") if st2.get("active_page") else None
            next_c = next(
                (
                    t
                    for t in sorted(st2["todo"], key=lambda x: x.get("order") or 0)
                    if (not active_pid or t.get("page_id") == active_pid) and t["status"] == "locked"
                ),
                None,
            ) or next((t for t in st2["todo"] if t["status"] == "locked"), None)
            if next_c:
                self.mem.set_concept_status(uid, next_c["id"], "learning")
                self.mem.set_current_concept(uid, next_c["id"], next_c["name"])
                self.mem.record_interaction(uid, next_c["name"], "examiner", "advanced")
                resp_ass = (
                    f"**Nice!** You passed _{concept_name}_ (score {score}/100).\n\n{feedback}\n\n"
                    f"---\n\n**Next up:** _{next_c['name']}_"
                )
                self.mem.add_message(uid, "assistant", resp_ass, session_id=state.get("session_id"), goal=state.get("goal"))
                self.mem.set_active_agent(uid, "instructor")
                return {
                    **state,
                    "score": score,
                    "feedback": feedback,
                    "response": resp_ass,
                    "next_agent": "instructor",
                    "concept_name": next_c["name"],
                    "concept_id": next_c["id"],
                    "retries": 0,
                }
            resp_ass = (
                f"**Mastered!** _{concept_name}_ (score {score}/100).\n\n{feedback}\n\n"
                f"---\n\nYou've completed this learning path. Want to set a new goal?"
            )
            self.mem.add_message(uid, "assistant", resp_ass, session_id=state.get("session_id"), goal=state.get("goal"))
            self.mem.end_session(uid, state.get("session_id"))
            self.mem.set_active_agent(uid, None)
            return {**state, "score": score, "feedback": feedback, "response": resp_ass, "next_agent": END}

        self.mem.record_interaction(uid, concept_name, "examiner", "needs_review", notes=f"score={score}")
        retries = int(state.get("retries") or 0) + 1
        if retries >= 2:
            # avoid infinite re-teach loop: advance anyway
            self.mem.set_concept_status(uid, concept_id, "mastered") if concept_id else None
            self.mem.set_current_concept(uid, None, None)
            st2 = self.mem.get_state(uid)
            active_pid = st2.get("active_page", {}).get("id") if st2.get("active_page") else None
            next_c = next(
                (
                    t
                    for t in sorted(st2["todo"], key=lambda x: x.get("order") or 0)
                    if (not active_pid or t.get("page_id") == active_pid) and t["status"] == "locked"
                ),
                None,
            ) or next((t for t in st2["todo"] if t["status"] == "locked"), None)
            if next_c:
                self.mem.set_concept_status(uid, next_c["id"], "learning")
                self.mem.set_current_concept(uid, next_c["id"], next_c["name"])
                resp_ass = (
                    f"Let's keep moving. I'll mark _{concept_name}_ for now and move to _{next_c['name']}_. "
                    f"You can revisit it anytime from the to-do list."
                )
                self.mem.add_message(uid, "assistant", resp_ass, session_id=state.get("session_id"), goal=state.get("goal"))
                return {
                    **state,
                    "score": score,
                    "feedback": feedback,
                    "response": resp_ass,
                    "next_agent": "instructor",
                    "concept_name": next_c["name"],
                    "concept_id": next_c["id"],
                    "retries": 0,
                }
            resp_ass = f"Let's pause here on _{concept_name}_. Want to set a new goal?"
            self.mem.add_message(uid, "assistant", resp_ass, session_id=state.get("session_id"), goal=state.get("goal"))
            self.mem.set_active_agent(uid, None)
            return {**state, "score": score, "feedback": feedback, "response": resp_ass, "next_agent": END}

        resp_ass = (
            f"**Not quite yet.** Score {score}/100 for _{concept_name}_.\n\n{feedback}\n\n"
            f"---\n\nI'll re-teach it a different way."
        )
        self.mem.add_message(uid, "assistant", resp_ass, session_id=state.get("session_id"), goal=state.get("goal"))
        self.mem.set_active_agent(uid, "instructor")
        return {
            **state,
            "score": score,
            "feedback": feedback,
            "response": resp_ass,
            "next_agent": "instructor",
            "retries": retries,
        }

    # ----------------------------------------------------------- build/run
    def build_graph(self):
        g = StateGraph(TutorState)
        g.add_node("planner", self.planner)
        g.add_node("linker", self.linker)
        g.add_node("instructor", self.instructor)
        g.add_node("examiner", self.examiner)
        from langgraph.graph import START

        g.add_conditional_edges(
            START,
            lambda s: s.get("next_agent") or "planner",
            {"planner": "planner", "instructor": "instructor", "examiner": "examiner"},
        )
        g.add_conditional_edges(
            "planner",
            lambda s: s.get("next_agent") or "linker",
            {"linker": "linker", "instructor": "instructor", END: END},
        )
        g.add_conditional_edges(
            "linker",
            lambda s: s.get("next_agent") or "instructor",
            {"instructor": "instructor", END: END},
        )
        g.add_conditional_edges(
            "instructor",
            lambda s: s.get("next_agent") or "examiner",
            {"examiner": "examiner", "instructor": "instructor", END: END},
        )
        g.add_conditional_edges(
            "examiner",
            lambda s: s.get("next_agent") or "instructor",
            {"instructor": "instructor", "examiner": "examiner", END: END},
        )
        return g.compile()

    def run_turn(
        self,
        user_id: str,
        message: str,
        goal: Optional[str] = None,
        is_reply: Optional[bool] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if session_id is None:
            # reuse open session if one exists
            st0 = self.mem.get_state(user_id)
            open_s = next((s for s in st0["sessions"] if s.get("ended_at") is None), None)
            session_id = open_s["id"] if open_s else self.mem.ensure_session(user_id, goal=goal)

        self.mem.add_message(user_id, "user", message, session_id=session_id, goal=goal)

        # Decide entry: grade an answer only if there's an active concept to grade
        # against AND the message isn't a brand-new goal declaration.
        import re

        goal_like = bool(
            re.match(
                r"^\s*(i\s*(want|'d\s*want|would\s*like)\s*to\s*learn|teach\s*me|"
                r"let'?s\s*learn|start\s*learning|help\s*me\s*learn|i'?d\s*like\s*to\s*learn|"
                r"new\s*goal|learn\s+about|i\s*need\s*to\s*learn)\b",
                message.strip(),
                re.IGNORECASE,
            )
        )
        if goal_like:
            # strip the intent prefix so the page is titled "dynamic programming"
            # instead of the whole sentence — and so the stale active-page goal
            # the frontend attaches doesn't overwrite the new one.
            cleaned = re.sub(
                r"^\s*(i\s*(want|'d\s*want|would\s*like)\s*to\s*learn|teach\s*me\s+|"
                r"let'?s\s*learn\s+|start\s*learning\s+(?:about\s+)?|help\s*me\s*learn\s+(?:about\s+)?|"
                r"i'?d\s*like\s*to\s*learn\s+|new\s*goal\s*:\s*|learn\s+about\s+|i\s*need\s*to\s*learn\s+)",
                "",
                message.strip(),
                flags=re.IGNORECASE,
            ).strip(" .!?,:")
            goal = cleaned or message.strip()

        st0 = self.mem.get_state(user_id)
        active_pid = st0.get("active_page", {}).get("id") if st0.get("active_page") else None
        cur = st0.get("current_concept")
        cur_on_active = bool(
            cur and active_pid and any(t["id"] == cur["id"] and t.get("page_id") == active_pid for t in st0["todo"])
        )
        entry = "examiner" if (cur_on_active and not goal_like) else "planner"

        init: TutorState = {
            "user_id": user_id,
            "message": message,
            "goal": goal,
            "is_reply": is_reply,
            "is_new_goal": goal_like,
            "session_id": session_id,
            "next_agent": entry,
            "retries": 0,
        }
        app = self.build_graph()
        out: TutorState = app.invoke(init, {"recursion_limit": 10})
        resp = out.get("response") or ""
        state_full = self.mem.get_state(user_id)
        return {
            "response": resp,
            "state": state_full,
            "next_agent": out.get("next_agent"),
            "score": out.get("score"),
            "feedback": out.get("feedback"),
        }
