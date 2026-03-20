import { useState, useEffect, useCallback, useRef } from "react";
import { useWs } from "@/hooks/use-ws";
import { useWsEvent } from "@/hooks/use-ws-event";
import { useAuthStore } from "@/stores/use-auth-store";
import { Methods, Events } from "@/api/protocol";
import type { Message } from "@/types/session";
import type { ChatMessage, AgentEventPayload, ToolStreamEntry } from "@/types/chat";

// --- sessionStorage persistence for in-flight streaming state ---
const STORAGE_PREFIX = "goclaw-stream:";

interface StreamSnapshot {
  messages: ChatMessage[];
  streamText: string | null;
  thinkingText: string | null;
  toolStream: ToolStreamEntry[];
  isRunning: boolean;
}

function saveStreamState(key: string, state: StreamSnapshot) {
  try { sessionStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(state)); } catch { /* quota */ }
}

function loadStreamState(key: string): StreamSnapshot | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_PREFIX + key);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function clearStreamState(key: string) {
  try { sessionStorage.removeItem(STORAGE_PREFIX + key); } catch { /* ignore */ }
}

/**
 * Manages chat message history and real-time streaming for a session.
 * Listens to "agent" events for chunks, tool calls, and run lifecycle.
 *
 * Streaming state is persisted to sessionStorage so navigating away
 * mid-stream and returning restores the in-progress view (the backend
 * only commits messages to the session store after run.completed).
 *
 * The runId is captured from the first "run.started" event (not from the
 * chat.send RPC response, which only arrives after the run completes).
 */
export function useChatMessages(sessionKey: string, agentId: string) {
  const ws = useWs();
  const connected = useAuthStore((s) => s.connected);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamText, setStreamText] = useState<string | null>(null);
  const [thinkingText, setThinkingText] = useState<string | null>(null);
  const [toolStream, setToolStream] = useState<ToolStreamEntry[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [loading, setLoading] = useState(false);

  // Use refs for values accessed inside the event handler to avoid stale closures.
  const runIdRef = useRef<string | null>(null);
  const expectingRunRef = useRef(false);
  const streamRef = useRef("");
  const thinkingRef = useRef("");
  const toolStreamRef = useRef<ToolStreamEntry[]>([]);
  const isRunningRef = useRef(false);
  isRunningRef.current = isRunning;
  const agentIdRef = useRef(agentId);
  agentIdRef.current = agentId;
  const loadHistoryRef = useRef<(() => Promise<void>) | null>(null);
  const pendingEventsRef = useRef<AgentEventPayload[]>([]);
  const messagesRef = useRef<ChatMessage[]>([]);
  messagesRef.current = messages;
  const sessionKeyRef = useRef(sessionKey);
  sessionKeyRef.current = sessionKey;
  const lastSaveRef = useRef(0);

  // Helper: persist current streaming state to sessionStorage.
  const persistSnapshot = useCallback(() => {
    const sk = sessionKeyRef.current;
    if (!sk) return;
    saveStreamState(sk, {
      messages: messagesRef.current,
      streamText: streamRef.current || null,
      thinkingText: thinkingRef.current || null,
      toolStream: toolStreamRef.current,
      isRunning: isRunningRef.current,
    });
  }, []);

  // Throttled persist for high-frequency events (chunks).
  const persistThrottled = useCallback(() => {
    const now = Date.now();
    if (now - lastSaveRef.current > 500) {
      lastSaveRef.current = now;
      persistSnapshot();
    }
  }, [persistSnapshot]);

  // Synchronously clear state during render when session changes.
  // This prevents a flash of old messages before the useEffect fires.
  // Then attempt to restore from sessionStorage if a run was in-flight.
  const [prevKey, setPrevKey] = useState(sessionKey);
  if (sessionKey !== prevKey) {
    setPrevKey(sessionKey);
    // NOTE: Do NOT reset expectingRunRef here. When a new chat is started,
    // handleSend() calls setSessionKey() (async) then send() which sets
    // expectingRunRef=true. The React re-render that applies the new sessionKey
    // fires AFTER expectRun() — resetting it here would kill the gate before
    // run.started arrives from the backend, causing all events to be dropped.
    runIdRef.current = null;
    streamRef.current = "";
    thinkingRef.current = "";
    toolStreamRef.current = [];
    // Keep pendingEventsRef — events may already be buffered for this new session.

    // Try to restore in-flight streaming state from sessionStorage
    const snapshot = loadStreamState(sessionKey);
    if (snapshot && snapshot.isRunning) {
      setMessages(snapshot.messages);
      setStreamText(snapshot.streamText);
      setThinkingText(snapshot.thinkingText);
      setToolStream(snapshot.toolStream);
      setIsRunning(true);
      setLoading(false);
      streamRef.current = snapshot.streamText ?? "";
      thinkingRef.current = snapshot.thinkingText ?? "";
      toolStreamRef.current = snapshot.toolStream;
    } else if (expectingRunRef.current) {
      // We just sent a message that created this session key — keep the
      // optimistic user message that addLocalMessage() already added.
      // Don't clear messages or set loading (the run is about to start).
      setStreamText(null);
      setThinkingText(null);
      setToolStream([]);
      setLoading(false);
    } else {
      setMessages([]);
      setStreamText(null);
      setThinkingText(null);
      setToolStream([]);
      setIsRunning(false);
      setLoading(true);
      clearStreamState(sessionKey);
    }
  }

  // Load history (no loading spinner — the empty state placeholder is shown instead)
  const loadHistory = useCallback(async () => {
    if (!sessionKey) {
      setLoading(false);
      return;
    }
    if (!ws.isConnected) {
      // Keep loading=true so the reconnect effect retriggers loadHistory.
      return;
    }
    try {
      const res = await ws.call<{ messages: Message[] }>(Methods.CHAT_HISTORY, {
        agentId,
        sessionKey,
      });
      const allMsgs = res.messages ?? [];
      // Build a map of tool_call_id -> tool message for result lookup
      const toolResultMap = new Map<string, Message>();
      for (const m of allMsgs) {
        if (m.role === "tool" && m.tool_call_id) {
          toolResultMap.set(m.tool_call_id, m);
        }
      }
      const msgs: ChatMessage[] = allMsgs.map((m: Message, i: number) => {
        const chatMsg: ChatMessage = {
          ...m,
          timestamp: Date.now() - (allMsgs.length - i) * 1000,
        };
        // Reconstruct toolDetails for assistant messages with tool_calls
        if (m.role === "assistant" && m.tool_calls && m.tool_calls.length > 0) {
          chatMsg.toolDetails = m.tool_calls.map((tc) => {
            const toolMsg = toolResultMap.get(tc.id);
            return {
              toolCallId: tc.id,
              runId: "",
              name: tc.name,
              phase: (toolMsg ? "completed" : "calling") as ToolStreamEntry["phase"],
              startedAt: 0,
              updatedAt: 0,
              arguments: tc.arguments,
              result: toolMsg?.content,
            };
          });
        }
        return chatMsg;
      });
      setMessages(msgs);
    } catch {
      // will retry
    } finally {
      setLoading(false);
    }
  }, [ws, agentId, sessionKey]);
  loadHistoryRef.current = loadHistory;

  // Load history when session changes
  useEffect(() => {
    if (sessionKey) {
      loadHistory();
    }
  }, [sessionKey, loadHistory]);

  // On WS reconnect: reset stuck streaming state and reload history.
  // Keep the sessionStorage snapshot so navigating away still restores it
  // (the backend run may still be in progress — history won't have the messages yet).
  useEffect(() => {
    if (!connected || !sessionKey) return;
    if (isRunningRef.current) {
      // Persist one last time before resetting, so the snapshot survives.
      persistSnapshot();
      setIsRunning(false);
      streamRef.current = "";
      setStreamText(null);
      setThinkingText(null);
      setToolStream([]);
      runIdRef.current = null;
      expectingRunRef.current = false;
      thinkingRef.current = "";
      toolStreamRef.current = [];
    }
    loadHistory();
  }, [connected, sessionKey, loadHistory, persistSnapshot]);

  // Called before sending a message so the event handler knows to capture run.started
  const expectRun = useCallback(() => {
    expectingRunRef.current = true;
  }, []);

  // Process a single event against the current run state (used by both
  // the live handler and the replay path after run.started arrives).
  const processRunEvent = useCallback((event: AgentEventPayload) => {
    switch (event.type) {
      case "thinking": {
        const content = event.payload?.content ?? "";
        thinkingRef.current += content;
        setThinkingText(thinkingRef.current);
        persistThrottled();
        break;
      }
      case "chunk": {
        const content = event.payload?.content ?? "";
        streamRef.current += content;
        setStreamText(streamRef.current);
        persistThrottled();
        break;
      }
      case "tool.call": {
        const entry: ToolStreamEntry = {
          toolCallId: event.payload?.id ?? "",
          runId: event.runId,
          name: event.payload?.name ?? "tool",
          arguments: event.payload?.arguments,
          phase: "calling",
          startedAt: Date.now(),
          updatedAt: Date.now(),
        };
        toolStreamRef.current = [...toolStreamRef.current, entry];
        setToolStream(toolStreamRef.current);
        persistSnapshot();
        break;
      }
      case "tool.result": {
        const isError = event.payload?.is_error;
        const resultId = event.payload?.id;
        const now = Date.now();
        toolStreamRef.current = toolStreamRef.current.map((t) =>
          t.toolCallId === resultId
            ? {
                ...t,
                phase: isError ? ("error" as const) : ("completed" as const),
                errorContent: isError ? event.payload?.content : undefined,
                result: event.payload?.result,
                updatedAt: now,
              }
            : t,
        );
        setToolStream(toolStreamRef.current);
        persistSnapshot();
        break;
      }
      case "run.completed": {
        setIsRunning(false);
        runIdRef.current = null;
        pendingEventsRef.current = [];

        const hadTools = toolStreamRef.current.length > 0;
        const streamed = streamRef.current;

        setStreamText(null);
        setThinkingText(null);
        setToolStream([]);
        streamRef.current = "";
        thinkingRef.current = "";
        toolStreamRef.current = [];

        if (streamed && !hadTools) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: streamed, timestamp: Date.now() },
          ]);
        } else {
          loadHistoryRef.current?.();
        }
        clearStreamState(sessionKeyRef.current);
        break;
      }
      case "run.failed": {
        setIsRunning(false);
        runIdRef.current = null;
        pendingEventsRef.current = [];
        setStreamText(null);
        setThinkingText(null);
        setToolStream([]);
        streamRef.current = "";
        thinkingRef.current = "";
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${event.payload?.error ?? "Unknown error"}`,
            timestamp: Date.now(),
          },
        ]);
        clearStreamState(sessionKeyRef.current);
        break;
      }
    }
  }, [persistSnapshot, persistThrottled]);

  // Adopt a runId from buffered events and replay them.
  const adoptRunId = useCallback((runId: string) => {
    runIdRef.current = runId;
    expectingRunRef.current = false;
    setIsRunning(true);
    setStreamText(null);
    setThinkingText(null);
    setToolStream([]);
    streamRef.current = "";
    thinkingRef.current = "";
    toolStreamRef.current = [];

    // Replay buffered events matching this runId
    const buffered = pendingEventsRef.current.filter((e) => e.runId === runId);
    pendingEventsRef.current = [];
    for (const e of buffered) {
      processRunEvent(e);
    }
    persistSnapshot();
  }, [processRunEvent, persistSnapshot]);

  // Stable event handler — reads everything from refs so deps are empty.
  // This ensures useWsEvent never unsubs/resubs mid-stream.
  const handleAgentEvent = useCallback(
    (payload: unknown) => {
      const event = payload as AgentEventPayload;
      if (!event) return;

      // Announce run.completed: reload history when a subagent/delegate announce finishes
      // for the current agent (these runs aren't tracked by runIdRef).
      if (event.type === "run.completed" && event.runKind === "announce" && event.agentId === agentIdRef.current) {
        loadHistoryRef.current?.();
        return;
      }

      // Capture run.started when we are expecting a run for this agent
      if (event.type === "run.started") {
        if (expectingRunRef.current && event.agentId === agentIdRef.current) {
          adoptRunId(event.runId);
        }
        return;
      }

      // If we have an active runId, process matching events directly
      if (runIdRef.current) {
        if (event.runId === runIdRef.current) {
          processRunEvent(event);
        }
        return;
      }

      // No active runId — buffer events if we are expecting a run
      if (expectingRunRef.current && event.agentId === agentIdRef.current && event.runId) {
        // Buffer the event (cap at 100 to prevent memory leak)
        if (pendingEventsRef.current.length < 100) {
          pendingEventsRef.current.push(event);
        }

        // Auto-adopt: if we have 3+ events with the same runId, assume
        // run.started was missed and adopt that runId.
        const candidateId = event.runId;
        const matchCount = pendingEventsRef.current.filter((e) => e.runId === candidateId).length;
        if (matchCount >= 3) {
          adoptRunId(candidateId);
        }
      }
    },
    [processRunEvent, adoptRunId],
  );

  useWsEvent(Events.AGENT, handleAgentEvent);

  // Add a local message optimistically (shown immediately, replaced on next loadHistory).
  // Persists to sessionStorage so the user's own message survives navigation.
  const addLocalMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => {
      const next = [...prev, msg];
      messagesRef.current = next;
      persistSnapshot();
      return next;
    });
  }, [persistSnapshot]);

  return {
    messages,
    streamText,
    thinkingText,
    toolStream,
    isRunning,
    loading,
    expectRun,
    loadHistory,
    addLocalMessage,
  };
}
