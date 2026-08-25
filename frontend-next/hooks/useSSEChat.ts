import { useState, useCallback, useEffect } from "react";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  isStreaming?: boolean;
}

const BACKEND_URL = "/api/proxy";

export function useSSEChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [sessionId, setSessionId] = useState<string>("");
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");

  // Initialize persistent session ID
  useEffect(() => {
    let sid = localStorage.getItem("pitwall_session_id");
    if (!sid) {
      sid = crypto.randomUUID();
      localStorage.setItem("pitwall_session_id", sid);
    }
    setSessionId(sid);
  }, []);

  // Health check polling
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/health`, { method: "GET" });
      if (res.ok) {
        setBackendStatus("online");
      } else {
        setBackendStatus("offline");
      }
    } catch {
      setBackendStatus("offline");
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Send message & stream response via SSE
  const sendMessage = useCallback(
    async (userPrompt: string) => {
      if (!userPrompt.trim() || isLoading) return;

      const userMsgId = crypto.randomUUID();
      const assistantMsgId = crypto.randomUUID();
      const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

      const userMessage: Message = {
        id: userMsgId,
        role: "user",
        content: userPrompt,
        timestamp: timeStr,
      };

      const initialAssistantMessage: Message = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        timestamp: timeStr,
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
      setIsLoading(true);

      // Log prompt to telemetry
      fetch("/api/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          level: "INFO",
          source: "Client",
          message: `User sent prompt (Length: ${userPrompt.length}) in session ${sessionId}`
        })
      }).catch(() => {});

      let retries = 0;
      const MAX_RETRIES = 3;

      const attemptFetch = async (): Promise<void> => {
        const token = sessionStorage.getItem("pitwall_jwt");
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (token) {
          headers["Authorization"] = `Bearer ${token}`;
        }

        let response = await fetch(`${BACKEND_URL}/chat`, {
          method: "POST",
          headers: headers,
          body: JSON.stringify({
            session_id: sessionId,
            message: userPrompt,
          }),
        });

        if (response.status === 401) {
          // Attempt silent refresh
          const refreshToken = localStorage.getItem("pitwall_refresh_token");
          if (refreshToken) {
            try {
              const refreshRes = await fetch(`${BACKEND_URL}/refresh`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ refresh_token: refreshToken })
              });
              
              if (refreshRes.ok) {
                const data = await refreshRes.json();
                sessionStorage.setItem("pitwall_jwt", data.access_token);
                localStorage.setItem("pitwall_refresh_token", data.refresh_token);
                
                // Retry the original request with new token
                const retryResponse = await fetch(`${BACKEND_URL}/chat`, {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${data.access_token}`
                  },
                  body: JSON.stringify({
                    session_id: sessionId,
                    message: userPrompt,
                  }),
                });
                
                if (!retryResponse.ok) throw new Error("Retry failed");
                response = retryResponse;
              } else {
                throw new Error("Refresh failed");
              }
            } catch (err) {
              sessionStorage.removeItem("pitwall_jwt");
              localStorage.removeItem("pitwall_refresh_token");
              window.location.href = "/login";
              return;
            }
          } else {
            sessionStorage.removeItem("pitwall_jwt");
            window.location.href = "/login";
            return;
          }
        }

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");

        if (!reader) throw new Error("Response body is null");

        let accumulatedContent = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const jsonStr = line.replace("data: ", "").trim();
              if (!jsonStr) continue;

              try {
                const parsed = JSON.parse(jsonStr);

                if (parsed.done) {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMsgId ? { ...msg, isStreaming: false } : msg
                    )
                  );
                  return; // Success exit
                }

                if (parsed.error) {
                  accumulatedContent += `\n\n⚠️ **Error:** ${parsed.error}`;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMsgId
                        ? { ...msg, content: accumulatedContent, isStreaming: false }
                        : msg
                    )
                  );
                  return; // Backend handled error
                }

                if (parsed.chunk) {
                  accumulatedContent += parsed.chunk;
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMsgId
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    )
                  );
                }
              } catch (e) {
                // Ignore parse errors on incomplete chunk splits
              }
            }
          }
        }
        
        throw new Error("Stream disconnected unexpectedly before completion.");
      };

      try {
        await attemptFetch();
      } catch (err: any) {
        if (retries < MAX_RETRIES) {
          retries++;
          const backoffMs = Math.pow(2, retries - 1) * 1000;
          
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? { ...msg, content: `⚠️ Connection dropped. Retrying (${retries}/${MAX_RETRIES})...` }
                : msg
            )
          );
          
          await new Promise(r => setTimeout(r, backoffMs));
          
          // Reset the message content for the retry
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? { ...msg, content: "" }
                : msg
            )
          );
          
          return attemptFetch(); // Recursive retry
        }

        // Log error to telemetry
        fetch("/api/log", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            level: "ERROR",
            source: "Client",
            message: `Connection Error: ${err.message} in session ${sessionId}`
          })
        }).catch(() => {});

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  content: `⚠️ **Connection Error:** Could not reach PitWall API backend. Please ensure Uvicorn server is running on port 8000.\n\n*Error details:* \`${err.message}\``,
                  isStreaming: false,
                }
              : msg
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, sessionId]
  );

  // Clear session
  const clearSession = useCallback(async () => {
    const token = sessionStorage.getItem("pitwall_jwt");
    if (sessionId) {
      try {
        const headers: Record<string, string> = {};
        if (token) {
          headers["Authorization"] = `Bearer ${token}`;
        }
        await fetch(`${BACKEND_URL}/session/${sessionId}`, { 
          method: "DELETE",
          headers: headers
        });
      } catch {
        // Ignore network errors on reset
      }
    }
    const newSid = crypto.randomUUID();
    localStorage.setItem("pitwall_session_id", newSid);
    setSessionId(newSid);
    setMessages([]);
  }, [sessionId]);

  return {
    messages,
    isLoading,
    backendStatus,
    sendMessage,
    clearSession,
  };
}
