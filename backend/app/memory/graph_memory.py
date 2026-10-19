from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from falkordb import FalkorDB
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.config import settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat() + "Z"


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# Labels/edge types that are memory plumbing, not part of the knowledge graph view.
HIDDEN_NODE_LABELS = {"User", "Message", "Session", "Interaction", "Strategy"}
HIDDEN_EDGE_TYPES = {"HAS_MESSAGE", "HAS_SESSION", "TAUGHT_BY", "DISCUSSED", "OPENED"}


class GraphMemory:
    """All FalkorDB reads/writes for one shared, per-user knowledge graph."""

    def __init__(self) -> None:
        # falkordb client accepts host/port or url
        try:
            self.db = FalkorDB.from_url(settings.falkordb_url)
        except Exception:
            self.db = FalkorDB(host="localhost", port=6379)

    def _graph(self, user_id: str):
        graph_name = f"tutor_{user_id}"
        return self.db.select_graph(graph_name)

    # ------------------------------------------------------------------ state
    def get_state(self, user_id: str) -> Dict[str, Any]:
        g = self._graph(user_id)
        self._ensure_user(g, user_id)

        # nodes — knowledge-graph view only (skip memory plumbing: messages,
        # sessions, interactions, strategies, the user node)
        nodes_res = g.query(
            "MATCH (n) RETURN n",
            {},
        )
        nodes: List[Dict[str, Any]] = []
        visible_ids: set = set()
        for rec in nodes_res.result_set:
            n = rec[0]
            if n is None:
                continue
            props = dict(n.properties)
            labels = list(n.labels)
            if set(labels) & HIDDEN_NODE_LABELS:
                continue
            nid = props.get("id") or str(n.id)
            visible_ids.add(nid)
            nodes.append(
                {
                    "id": nid,
                    "labels": labels,
                    "properties": props,
                }
            )

        # edges — only between visible nodes, skip memory-plumbing edge types
        edges_res = g.query(
            "MATCH (a)-[r]->(b) "
            "RETURN a.id AS s, b.id AS t, TYPE(r) AS ty, r AS rel",
            {},
        )
        edges: List[Dict[str, Any]] = []
        seen = set()
        for rec in edges_res.result_set:
            src_id, dst_id, rtype, rel = rec[0], rec[1], rec[2], rec[3]
            if src_id is None or dst_id is None:
                continue
            if src_id not in visible_ids or dst_id not in visible_ids:
                continue
            if rtype in HIDDEN_EDGE_TYPES:
                continue
            rid = f"{src_id}->{rtype}->{dst_id}"
            if rid in seen:
                continue
            seen.add(rid)
            rel_props = dict(rel.properties) if rel else {}
            edges.append(
                {
                    "id": rid,
                    "source": src_id,
                    "target": dst_id,
                    "type": rtype,
                    "properties": rel_props,
                }
            )

        # pages
        pages_res = g.query(
            "MATCH (p:Page) RETURN p ORDER BY p.created_at ASC",
            {},
        )
        pages: List[Dict[str, Any]] = []
        for rec in pages_res.result_set:
            p = rec[0]
            if not p:
                continue
            pp = dict(p.properties)
            pages.append(
                {
                    "id": pp.get("id"),
                    "goal": pp.get("goal"),
                    "title": pp.get("title") or pp.get("goal"),
                    "active": bool(pp.get("active")),
                    "created_at": pp.get("created_at"),
                }
            )

        # active page
        active_res = g.query(
            "MATCH (p:Page {active: true}) RETURN p.id AS id, p.goal AS goal LIMIT 1",
            {},
        )
        active_page = None
        if active_res.result_set:
            aid, agoal = active_res.result_set[0]
            active_page = {"id": aid, "goal": agoal}

        # todo — concepts can belong to MULTIPLE goals (shared nodes). Each item
        # carries every page it's on plus that page's learning order, so the UI
        # can group it under each goal without duplicating the graph node.
        apid = active_page["id"] if active_page else None
        concepts_res = g.query(
            "MATCH (c:Concept) RETURN c ORDER BY c.order ASC, c.created_at ASC",
            {},
        )
        link_res = g.query(
            "MATCH (p:Page)-[r:HAS_CONCEPT]->(c:Concept) RETURN p.id, c.id, r.order, p.created_at",
            {},
        )
        pages_meta = {p["id"]: (p.get("created_at") or "") for p in pages}
        membership: Dict[str, Dict[str, Any]] = {}
        for pid, cid, rorder, pcreated in link_res.result_set:
            if not cid:
                continue
            m = membership.setdefault(cid, {"page_ids": [], "orders": {}})
            if pid and pid not in m["page_ids"]:
                m["page_ids"].append(pid)
            if pid and rorder is not None:
                m["orders"][pid] = int(rorder)
        # stable page order: active first, then oldest page first
        def _page_sort_key(pid: str) -> tuple:
            return (0 if pid == apid else 1, pages_meta.get(pid, ""), pid)

        todo: List[Dict[str, Any]] = []
        for rec in concepts_res.result_set:
            cnode = rec[0]
            if not cnode:
                continue
            cp = dict(cnode.properties)
            cid = cp.get("id")
            m = membership.get(cid, {"page_ids": [], "orders": {}})
            page_ids = sorted(m["page_ids"], key=_page_sort_key)
            page_id = page_ids[0] if page_ids else None
            orders = m["orders"]
            primary_order = orders.get(page_id, cp.get("order") or 0) if page_id else (cp.get("order") or 0)
            todo.append(
                {
                    "id": cid,
                    "name": cp.get("name"),
                    "status": cp.get("status") or "locked",
                    "explanation": cp.get("explanation"),
                    "description": cp.get("description"),
                    "order": primary_order,
                    "page_id": page_id,
                    "page_ids": page_ids,
                    "orders": orders,
                    "mastered_at": cp.get("mastered_at"),
                    "learning_at": cp.get("learning_at"),
                }
            )
        todo.sort(key=lambda t: (t["order"], t.get("name") or ""))

        # chat history
        chat_res = g.query(
            "MATCH (m:Message) RETURN m ORDER BY m.timestamp ASC LIMIT 500",
            {},
        )
        chat_history: List[Dict[str, Any]] = []
        for rec in chat_res.result_set:
            mnode = rec[0]
            if not mnode:
                continue
            mp = dict(mnode.properties)
            chat_history.append(
                {
                    "id": mp.get("id"),
                    "role": mp.get("role"),
                    "content": mp.get("content"),
                    "timestamp": mp.get("timestamp"),
                    "session_id": mp.get("session_id"),
                    "goal": mp.get("goal"),
                }
            )

        # sessions
        sess_res = g.query(
            "MATCH (s:Session) RETURN s.id, s.started_at, s.ended_at, s.goal ORDER BY s.started_at DESC",
            {},
        )
        sessions = [
            {"id": r[0], "started_at": r[1], "ended_at": r[2], "goal": r[3]}
            for r in sess_res.result_set
        ]

        # user meta
        aa_res = g.query(
            "MATCH (u:User {id:$uid}) RETURN u.active_agent, u.current_concept_id, u.current_concept_name",
            {"uid": user_id},
        )
        active_agent = None
        current_concept = None
        if aa_res.result_set:
            ag, cid, cname = aa_res.result_set[0]
            active_agent = ag
            if cid:
                current_concept = {"id": cid, "name": cname}

        return {
            "user_id": user_id,
            "graph": {"nodes": nodes, "edges": edges},
            "pages": pages,
            "active_page": active_page,
            "todo": todo,
            "chat_history": chat_history,
            "sessions": sessions,
            "active_agent": active_agent,
            "current_concept": current_concept,
        }

    # --------------------------------------------------------------- user/txn
    def _ensure_user(self, g, user_id: str) -> None:
        g.query(
            "MERGE (u:User {id: $uid}) "
            "ON CREATE SET u.created_at = $ts, u.active_agent = null, "
            "u.current_concept_id = null, u.current_concept_name = null",
            {"uid": user_id, "ts": _now_iso()},
        )

    def ensure_session(self, user_id: str, goal: Optional[str] = None) -> str:
        sid = _gen_id("sess")
        g = self._graph(user_id)
        self._ensure_user(g, user_id)
        g.query(
            "CREATE (s:Session {id: $sid, user_id: $uid, started_at: $ts, ended_at: null, goal: $goal})",
            {"sid": sid, "uid": user_id, "ts": _now_iso(), "goal": goal or ""},
        )
        return sid

    def end_session(self, user_id: str, session_id: Optional[str] = None) -> None:
        g = self._graph(user_id)
        ts = _now_iso()
        if session_id:
            g.query(
                "MATCH (s:Session {id: $sid}) SET s.ended_at = $ts",
                {"sid": session_id, "ts": ts},
            )
        else:
            g.query(
                "MATCH (s:Session {ended_at: null}) SET s.ended_at = $ts",
                {"ts": ts},
            )

    # --------------------------------------------------------------- messages
    def add_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> str:
        mid = _gen_id("msg")
        g = self._graph(user_id)
        self._ensure_user(g, user_id)
        g.query(
            "CREATE (m:Message {id: $mid, user_id: $uid, role: $role, content: $content, "
            "timestamp: $ts, session_id: $sid, goal: $goal})",
            {
                "mid": mid,
                "uid": user_id,
                "role": role,
                "content": content,
                "ts": _now_iso(),
                "sid": session_id or "",
                "goal": goal or "",
            },
        )
        return mid

    def get_chat_messages(self, user_id: str, limit: int = 100) -> List[Any]:
        g = self._graph(user_id)
        res = g.query(
            "MATCH (m:Message) RETURN m ORDER BY m.timestamp ASC LIMIT $lim",
            {"lim": limit},
        )
        msgs: List[Any] = []
        for rec in res.result_set:
            m = rec[0]
            if not m:
                continue
            mp = dict(m.properties)
            r = mp.get("role") or "assistant"
            c = mp.get("content") or ""
            if r == "user":
                msgs.append(HumanMessage(content=c))
            elif r == "system":
                msgs.append(SystemMessage(content=c))
            else:
                msgs.append(AIMessage(content=c))
        return msgs

    # ------------------------------------------------------------------ pages
    def create_page(self, user_id: str, goal: str, title: Optional[str] = None) -> str:
        pid = _gen_id("page")
        g = self._graph(user_id)
        self._ensure_user(g, user_id)
        g.query("MATCH (p:Page) SET p.active = false", {})
        g.query(
            "CREATE (p:Page {id: $pid, user_id: $uid, goal: $goal, title: $title, active: true, created_at: $ts})",
            {"pid": pid, "uid": user_id, "goal": goal, "title": title or goal, "ts": _now_iso()},
        )
        return pid

    def set_active_page(self, user_id: str, page_id: str) -> None:
        g = self._graph(user_id)
        g.query(
            "MATCH (p:Page) SET p.active = (p.id = $pid)",
            {"pid": page_id},
        )

    def activate_page(self, user_id: str, page_id: str) -> None:
        """Switch the active goal. If the current concept belongs to another
        goal, clear it so the planner re-targets on the next message."""
        g = self._graph(user_id)
        self.set_active_page(user_id, page_id)
        self._ensure_user(g, user_id)
        cur = g.query(
            "MATCH (u:User {id: $uid}) RETURN u.current_concept_id",
            {"uid": user_id},
        )
        cid = cur.result_set[0][0] if cur.result_set else None
        if not cid:
            return
        on_page = g.query(
            "MATCH (c:Concept {id: $cid})-[:HAS_CONCEPT]-(p:Page {id: $pid}) RETURN count(*)",
            {"cid": cid, "pid": page_id},
        )
        count = on_page.result_set[0][0] if on_page.result_set else 0
        if count == 0:
            g.query(
                "MATCH (u:User {id: $uid}) "
                "SET u.current_concept_id = null, u.current_concept_name = null",
                {"uid": user_id},
            )

    # --------------------------------------------------------------- concepts
    def upsert_concept(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        explanation: Optional[str] = None,
        order: int = 0,
        status: str = "locked",
        page_id: Optional[str] = None,
        page_order: Optional[int] = None,
    ) -> str:
        cid = _gen_id("c")
        g = self._graph(user_id)
        self._ensure_user(g, user_id)
        # MERGE on (user_id, name): the SAME concept node is reused whenever a
        # name matches — one node per idea, no duplicates, status preserved.
        g.query(
            "MERGE (c:Concept {user_id: $uid, name: $name}) "
            "ON CREATE SET c.id = $cid, c.description = $desc, c.explanation = $expl, "
            "c.order = $order, c.status = $status, c.created_at = $ts, "
            "c.learning_at = null, c.mastered_at = null "
            "ON MATCH SET c.description = COALESCE($desc, c.description), "
            "c.explanation = COALESCE($expl, c.explanation), c.order = $order",
            {
                "uid": user_id,
                "name": name.strip(),
                "cid": cid,
                "desc": description or None,
                "expl": explanation or None,
                "order": order,
                "status": status,
                "ts": _now_iso(),
            },
        )
        fres = g.query(
            "MATCH (c:Concept {user_id:$uid, name:$name}) RETURN c.id",
            {"uid": user_id, "name": name.strip()},
        )
        real_cid = fres.result_set[0][0] if fres.result_set else cid
        if page_id:
            # per-page position lives on the edge, so a concept shared by two
            # goals keeps a different spot in each goal's to-do
            g.query(
                "MATCH (p:Page {id:$pid}), (c:Concept {id:$cid}) "
                "MERGE (p)-[r:HAS_CONCEPT]->(c) "
                "SET r.order = COALESCE($porder, r.order)",
                {"pid": page_id, "cid": real_cid, "porder": page_order},
            )
        return real_cid

    def set_concept_status(self, user_id: str, concept_id: str, status: str) -> None:
        if status not in ("locked", "learning", "mastered"):
            return
        ts = _now_iso()
        g = self._graph(user_id)
        g.query(
            "MATCH (c:Concept {id:$cid}) SET c.status = $status",
            {"cid": concept_id, "status": status},
        )
        if status == "learning":
            g.query("MATCH (c:Concept {id:$cid}) SET c.learning_at = $ts", {"cid": concept_id, "ts": ts})
        if status == "mastered":
            g.query("MATCH (c:Concept {id:$cid}) SET c.mastered_at = $ts", {"cid": concept_id, "ts": ts})

    def get_concept_by_name(self, user_id: str, name: str) -> Optional[Dict[str, Any]]:
        g = self._graph(user_id)
        r = g.query("MATCH (c:Concept {user_id:$uid, name:$name}) RETURN c", {"uid": user_id, "name": name})
        if not r.result_set:
            return None
        return dict(r.result_set[0][0].properties)

    def set_current_concept(
        self, user_id: str, concept_id: Optional[str], concept_name: Optional[str]
    ) -> None:
        g = self._graph(user_id)
        self._ensure_user(g, user_id)
        g.query(
            "MATCH (u:User {id:$uid}) SET u.current_concept_id = $cid, u.current_concept_name = $cname",
            {"uid": user_id, "cid": concept_id, "cname": concept_name},
        )

    # ------------------------------------------------------------- strategies
    def record_taught_strategy(
        self, user_id: str, concept_name: str, strategy: str, success: bool = False
    ) -> None:
        sid = _gen_id("str")
        g = self._graph(user_id)
        self.upsert_concept(user_id, concept_name)
        g.query(
            "CREATE (s:Strategy {id:$sid, user_id:$uid, strategy:$strat, success:$ok, created_at:$ts})",
            {"sid": sid, "uid": user_id, "strat": strategy, "ok": success, "ts": _now_iso()},
        )
        g.query(
            "MATCH (c:Concept {user_id:$uid, name:$name}), (st:Strategy {id:$sid}) "
            "MERGE (c)-[:TAUGHT_BY]->(st)",
            {"uid": user_id, "name": concept_name, "sid": sid},
        )

    def mark_strategy_success(self, user_id: str, concept_name: str) -> None:
        g = self._graph(user_id)
        g.query(
            "MATCH (c:Concept {user_id:$uid, name:$name})-[r:TAUGHT_BY]->(st:Strategy) "
            "SET st.success = true",
            {"uid": user_id, "name": concept_name},
        )

    def get_successful_strategies(self, user_id: str, concept_name: str) -> List[str]:
        g = self._graph(user_id)
        r = g.query(
            "MATCH (c:Concept {user_id:$uid, name:$name})-[:TAUGHT_BY]->(st:Strategy {success:true}) "
            "RETURN DISTINCT st.strategy ORDER BY st.created_at DESC",
            {"uid": user_id, "name": concept_name},
        )
        return [x[0] for x in r.result_set if x[0]]

    # ------------------------------------------------------------ graph edits
    def add_node(self, user_id: str, labels: List[str], properties: Dict[str, Any]) -> str:
        nid_prop = properties.get("id") or _gen_id("n")
        props = dict(properties)
        props["id"] = nid_prop
        if "user_id" not in props:
            props["user_id"] = user_id
        lbls = ":".join(l for l in labels if l) or "Concept"
        g = self._graph(user_id)
        parts: List[str] = []
        params: Dict[str, Any] = {}
        for k, v in props.items():
            pk = f"p_{k}"
            parts.append(f"{k}: ${pk}")
            params[pk] = v
        q = f"CREATE (n:{lbls} {{{', '.join(parts)}}}) RETURN n.id AS id"
        res = g.query(q, params)
        return res.result_set[0][0] if res.result_set else nid_prop

    def delete_node(self, user_id: str, node_id: str) -> bool:
        g = self._graph(user_id)
        g.query("MATCH (n {id:$nid}) DETACH DELETE n", {"nid": node_id})
        return True

    def add_edge(
        self,
        user_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        g = self._graph(user_id)
        props = properties or {}
        parts: List[str] = []
        params: Dict[str, Any] = {"s": source_id, "t": target_id}
        for k, v in props.items():
            pk = f"p_{k}"
            parts.append(f"{k}: ${pk}")
            params[pk] = v
        propstr = "{" + ", ".join(parts) + "}" if parts else ""
        safe_type = "".join(ch for ch in edge_type if ch.isalnum() or ch == "_") or "REQUIRES"
        q = (
            f"MATCH (a {{id:$s}}), (b {{id:$t}}) "
            f"MERGE (a)-[r:{safe_type} {propstr}]->(b) RETURN id(r)"
        )
        g.query(q, params)
        return True

    def delete_edge(self, user_id: str, source_id: str, target_id: str, edge_type: str) -> bool:
        g = self._graph(user_id)
        safe_type = "".join(ch for ch in edge_type if ch.isalnum() or ch == "_") or "REQUIRES"
        g.query(
            f"MATCH (a {{id:$s}})-[r:{safe_type}]->(b {{id:$t}}) DELETE r",
            {"s": source_id, "t": target_id},
        )
        return True

    def save_positions(self, user_id: str, positions: List[Dict[str, Any]]) -> None:
        g = self._graph(user_id)
        for p in positions:
            nid = p.get("id")
            x = p.get("x")
            y = p.get("y")
            if nid is None or x is None or y is None:
                continue
            g.query(
                "MATCH (n {id:$nid}) SET n.x = $x, n.y = $y",
                {"nid": nid, "x": float(x), "y": float(y)},
            )

    # --------------------------------------------------------------- analyze
    def analyze_links(self, user_id: str) -> List[Dict[str, Any]]:
        g = self._graph(user_id)
        cres = g.query(
            "MATCH (c:Concept) RETURN c.id, c.name, c.description, c.explanation",
            {},
        )
        concepts = [
            {"id": r[0], "name": r[1], "description": r[2] or "", "explanation": r[3] or ""}
            for r in cres.result_set
        ]
        if len(concepts) < 2:
            return []
        if not settings.groq_api_key:
            return []
        try:
            llm = ChatGroq(api_key=settings.groq_api_key, model=settings.model, temperature=0.1)
        except Exception:
            return []

        schema_hint = (
            "Infer prerequisite/dependency links between concepts. Return ONLY a valid JSON array: "
            '[{"source_name":"...","target_name":"...","edge_type":"REQUIRES","properties":{}}] '
            "edge_type should be REQUIRES (preferred). source_name is the dependent concept, "
            "target_name is the prerequisite it requires. Use names exactly as given."
        )
        prompt = f"{schema_hint}\n\nConcepts:\n{json.dumps(concepts, indent=2)}"
        try:
            resp = llm.invoke(prompt)
            txt = resp.content if hasattr(resp, "content") else str(resp)
            start = txt.find("[")
            end = txt.rfind("]")
            if start == -1 or end == -1:
                return []
            data = json.loads(txt[start : end + 1])
        except Exception:
            return []

        created: List[Dict[str, Any]] = []
        for e in data or []:
            try:
                sn = e.get("source_name")
                tn = e.get("target_name")
                if not sn or not tn or sn == tn:
                    continue
                sres = g.query("MATCH (c:Concept {user_id:$uid, name:$sn}) RETURN c.id", {"uid": user_id, "sn": sn})
                tres = g.query("MATCH (c:Concept {user_id:$uid, name:$tn}) RETURN c.id", {"uid": user_id, "tn": tn})
                if not sres.result_set or not tres.result_set:
                    continue
                sidn, tidn = sres.result_set[0][0], tres.result_set[0][0]
                self.add_edge(user_id, sidn, tidn, "REQUIRES", e.get("properties") or {})
                created.append({"source": sidn, "target": tidn, "type": "REQUIRES"})
            except Exception:
                continue
        return created

    # ----------------------------------------------------------------- reset
    def reset(self, user_id: str) -> None:
        g = self._graph(user_id)
        g.query("MATCH (n) DETACH DELETE n", {})
        self._ensure_user(g, user_id)

    # -------------------------------------------------------------- agent meta
    def set_active_agent(self, user_id: str, agent: Optional[str]) -> None:
        g = self._graph(user_id)
        self._ensure_user(g, user_id)
        g.query("MATCH (u:User {id:$uid}) SET u.active_agent = $ag", {"uid": user_id, "ag": agent})

    def record_interaction(
        self,
        user_id: str,
        concept_name: Optional[str],
        agent: str,
        action: str,
        notes: Optional[str] = None,
    ) -> None:
        iid = _gen_id("int")
        g = self._graph(user_id)
        g.query(
            "CREATE (i:Interaction {id:$iid, user_id:$uid, concept_name:$cname, "
            "agent:$agent, action:$action, notes:$notes, created_at:$ts})",
            {
                "iid": iid,
                "uid": user_id,
                "cname": concept_name or "",
                "agent": agent,
                "action": action,
                "notes": notes or "",
                "ts": _now_iso(),
            },
        )
