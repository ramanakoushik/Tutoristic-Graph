"use client";

import { useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeMouseHandler,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";
import dagre from "@dagrejs/dagre";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Plus, Trash2, Link2 } from "lucide-react";
import { GraphNode, GraphEdge } from "@/lib/api";

interface GraphViewProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onSavePositions: (positions: { id: string; x: number; y: number }[]) => void;
  onAddNode: (labels: string[], properties: Record<string, any>) => void;
  onDeleteNode: (nodeId: string) => void;
  onAddEdge: (sourceId: string, targetId: string, edgeType: string, properties?: Record<string, any>) => void;
  onDeleteEdge: (sourceId: string, targetId: string, edgeType: string) => void;
}

const nodeWidth = 180;
const nodeHeight = 60;

const getLayout = (ns: any[], es: any[]) => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  // better spacing for cleaner, less cramped layout
  dagreGraph.setGraph({
    rankdir: "TB",
    ranksep: 100,
    nodesep: 80,
    marginx: 20,
    marginy: 20,
  });
  ns.forEach((n) => dagreGraph.setNode(n.id, { width: nodeWidth, height: nodeHeight }));
  // flip edges so prerequisites (REQUIRES targets) rank above dependents
  es.forEach((e) => dagreGraph.setEdge(e.target, e.source));
  dagre.layout(dagreGraph);
  return ns.map((n) => {
    const pos = dagreGraph.node(n.id);
    return {
      ...n,
      position: { x: pos.x - nodeWidth / 2, y: pos.y - nodeHeight / 2 },
    };
  });
};

function GraphInner({
  nodes,
  edges,
  onSavePositions,
  onAddNode,
  onDeleteNode,
  onAddEdge,
  onDeleteEdge,
}: GraphViewProps) {
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState([]);
  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState("");
  const [newLabel, setNewLabel] = useState("Concept");
  const [selected, setSelected] = useState<{ nodeId?: string; edgeId?: string }>({});
  const [edgeType, setEdgeType] = useState("REQUIRES");
  const [connectMode, setConnectMode] = useState(false);

  const mappedNodes = useMemo<any[]>(() => {
    return nodes.map((n) => {
      const props = n.properties || {};
      const name = props.name || props.title || props.goal || props.id;
      const status = props.status;
      let bg = "#f8fafc";
      let border = "#cbd5e1";
      if (status === "mastered") {
        bg = "#dcfce7";
        border = "#22c55e";
      }
      if (status === "learning") {
        bg = "#fef3c7";
        border = "#f59e0b";
      }
      if (status === "locked") {
        bg = "#f1f5f9";
        border = "#94a3b8";
      }
      const labelsStr = n.labels.join(", ");
      return {
        id: props.id || n.id,
        type: "default",
        data: {
          label: (
            <div className="flex flex-col">
              <span className="text-xs font-medium">{name}</span>
              {labelsStr && <span className="text-[9px] opacity-60">{labelsStr}</span>}
            </div>
          ),
        },
        position: { x: props.x ?? 0, y: props.y ?? 0 },
        style: {
          background: bg,
          border: `2px solid ${border}`,
          borderRadius: 8,
          padding: 8,
          width: nodeWidth,
          height: nodeHeight,
        },
      };
    });
  }, [nodes]);

  const mappedEdges = useMemo<any[]>(() => {
    return edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.type,
      type: "smoothstep",
      animated: false,
      style: { stroke: "#64748b" },
      labelStyle: { fontSize: 10 },
    }));
  }, [edges]);

  useEffect(() => {
    const hasPositions = mappedNodes.some((n: any) => n.position.x !== 0 || n.position.y !== 0);
    if (hasPositions) {
      setRfNodes(mappedNodes);
    } else {
      setRfNodes(getLayout(mappedNodes, mappedEdges));
    }
    setRfEdges(mappedEdges);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges]);

  const onConnect = (connection: Connection) => {
    if (!connection.source || !connection.target) return;
    if (connectMode) {
      onAddEdge(connection.source, connection.target, edgeType);
      setConnectMode(false);
    } else {
      setRfEdges((eds: any) => addEdge({ ...connection, type: "smoothstep", label: edgeType }, eds));
    }
  };

  const onNodeClick: NodeMouseHandler = (_e, node) => {
    setSelected({ nodeId: node.id });
  };

  const onEdgeClick = (_e: any, edge: Edge) => {
    setSelected({ edgeId: edge.id });
  };

  const onPaneClick = () => {
    setSelected({});
    if (connectMode) setConnectMode(false);
  };

  const handleAddNode = () => {
    if (!newName.trim()) return;
    onAddNode([newLabel], { name: newName.trim(), status: "locked" });
    setNewName("");
    setShowAdd(false);
  };

  const handleDeleteSelected = () => {
    if (selected.nodeId) onDeleteNode(selected.nodeId);
    if (selected.edgeId) {
      const e = rfEdges.find((x: any) => x.id === selected.edgeId);
      if (e && e.source && e.target) onDeleteEdge(e.source, e.target, (e.label as string) || "REQUIRES");
    }
    setSelected({});
  };

  const handleSavePositions = () => {
    onSavePositions(rfNodes.map((n: any) => ({ id: n.id, x: n.position.x, y: n.position.y })));
  };

  return (
    <div className="relative h-full w-full">
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={1.5}
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>

      <Card className="absolute left-4 top-4 w-64">
        <CardContent className="space-y-2 p-3">
          {!showAdd ? (
            <Button size="sm" variant="outline" className="w-full" onClick={() => setShowAdd(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Node
            </Button>
          ) : (
            <div className="space-y-2">
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Node name"
              />
              <Input
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                placeholder="Label (Concept/Page/User)"
              />
              <div className="flex gap-2">
                <Button size="sm" onClick={handleAddNode} disabled={!newName.trim()}>
                  Add
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setShowAdd(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          )}
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant={connectMode ? "default" : "outline"}
              onClick={() => setConnectMode((v) => !v)}
            >
              <Link2 className="mr-2 h-4 w-4" />
              {connectMode ? "Connecting..." : "Connect"}
            </Button>
            <Input value={edgeType} onChange={(e) => setEdgeType(e.target.value)} placeholder="REQUIRES" />
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" onClick={handleSavePositions}>
              Save Positions
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={handleDeleteSelected}
              disabled={!selected.nodeId && !selected.edgeId}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </Button>
          </div>
          {selected.nodeId && (
            <div className="text-xs text-muted-foreground">Selected node: {selected.nodeId}</div>
          )}
          {selected.edgeId && (
            <div className="text-xs text-muted-foreground">Selected edge: {selected.edgeId}</div>
          )}
          {connectMode && (
            <div className="text-xs text-muted-foreground">
              Drag from source node handle to target to create {edgeType}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function GraphView(props: GraphViewProps) {
  return (
    <ReactFlowProvider>
      <GraphInner {...props} />
    </ReactFlowProvider>
  );
}
