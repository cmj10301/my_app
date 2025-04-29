"use client";

import { useState } from "react";
import axios from "axios";

type Step =
  | { type: "question"; text: string; index: number }
  | { type: "guess"; name: string };

export default function TwentyQuestions() {
  const [sessionId, setSessionId] = useState<string>();
  const [step, setStep] = useState<Step>();
  const [history, setHistory] = useState<string[]>([]);

  const start = async () => {
    const res = await axios.post("http://localhost:8000/start");
    setSessionId(res.data.session_id);
    setStep({ type: "question", text: res.data.question, index: res.data.question_index });
    setHistory([]);
  };

  const sendAnswer = async (ans: 1 | 0 | -1) => {
    if (!sessionId || !step || step.type !== "question") return;

    const res = await axios.post("http://localhost:8000/answer", {
      session_id: sessionId,
      answer: ans,
    });

    // 기록
    setHistory((h) => [...h, `${step.text} → ${ans === 1 ? "예" : ans === 0 ? "아니오" : "모르겠습니다"}`]);

    if (res.data.guess) {
      setStep({ type: "guess", name: res.data.guess_name });
    } else {
      setStep({ type: "question", text: res.data.question, index: res.data.question_index });
    }
  };

  return (
    <main className="flex flex-col items-center gap-4 p-8">
      <h1 className="text-3xl font-bold">20 Questions – 동물 맞히기</h1>

      {!step && (
        <button className="px-6 py-3 rounded-lg shadow" onClick={start}>
          게임 시작
        </button>
      )}

      {step?.type === "question" && (
        <div className="space-y-4 text-center">
          <p className="text-xl">{step.text}</p>
          <div className="flex gap-2 justify-center">
            <button onClick={() => sendAnswer(1)} className="btn">
              예
            </button>
            <button onClick={() => sendAnswer(0)} className="btn">
              아니오
            </button>
            <button onClick={() => sendAnswer(-1)} className="btn">
              모르겠음
            </button>
          </div>
        </div>
      )}

      {step?.type === "guess" && (
        <div className="space-y-4 text-center">
          <p className="text-2xl font-semibold">제가 생각한 동물은 …</p>
          <p className="text-4xl">{step.name}</p>
          <button onClick={start} className="btn">
            다시 하기
          </button>
        </div>
      )}

      {history.length > 0 && (
        <div className="w-full max-w-md mt-6">
          <h2 className="font-medium mb-2">질문 / 답 기록</h2>
          <ul className="list-disc pl-5 text-sm space-y-1">
            {history.map((h, i) => (
              <li key={i}>{h}</li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}

/* tailwind btn */
const btn = "px-4 py-2 rounded-lg border shadow";
