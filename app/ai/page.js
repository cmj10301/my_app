"use client";

import { useState } from "react";
import axios from "axios";

export default function GamePage() {
  const [sessionId, setSessionId] = useState<string>("");
  const [question, setQuestion] = useState<string>("게임을 시작해주세요!");
  const [guess, setGuess] = useState<string>("");

  const startGame = async () => {
    const res = await axios.post("http://localhost:8000/start");
    setSessionId(res.data.session_id);
    setQuestion(res.data.question);
    setGuess("");
  };

  const sendAnswer = async (answer) => {
    if (!sessionId) {
      alert("게임을 먼저 시작하세요!");
      return;
    }
    const res = await axios.post("http://localhost:8000/answer", {
      session_id: sessionId,
      answer: answer,
    });

    if (res.data.guess) {
      setGuess(`정답 추측: ${res.data.guess_name}`);
      setQuestion("게임 종료!");
    } else {
      setQuestion(res.data.question);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4 p-4">
      <h1 className="text-2xl font-bold">스무고개 AI 게임</h1>
      <button onClick={startGame} className="bg-blue-500 text-white px-4 py-2 rounded">
        게임 시작
      </button>
      <div className="text-lg mt-6">{question}</div>
      <div className="flex gap-4 mt-4">
        <button onClick={() => sendAnswer(1)} className="bg-green-500 text-white px-4 py-2 rounded">
          예
        </button>
        <button onClick={() => sendAnswer(0)} className="bg-red-500 text-white px-4 py-2 rounded">
          아니오
        </button>
        <button onClick={() => sendAnswer(-1)} className="bg-gray-500 text-white px-4 py-2 rounded">
          모르겠습니다
        </button>
      </div>
      {guess && (
        <div className="mt-6 text-lg font-semibold">{guess}</div>
      )}
    </div>
  );
}
