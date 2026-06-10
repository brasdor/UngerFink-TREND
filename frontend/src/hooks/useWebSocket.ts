"use client";

import { useEffect, useRef, useCallback, useState } from "react";

type WebSocketMessage = {
  type: string;
  data?: any;
};

export function useWebSocket(url?: string) {
  const wsUrl =
    url || process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const connect = useCallback(() => {
    const ws = new WebSocket(`${wsUrl}/ws/live`);

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect after 5s
      setTimeout(connect, 5000);
    };
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        setLastMessage(msg);
      } catch {}
    };

    wsRef.current = ws;
  }, [wsUrl]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((msg: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { isConnected, lastMessage, sendMessage };
}
