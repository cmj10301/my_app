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

  /* ---------- helpers ---------- */
  const start = async () => {
    const { data } = await axios.post("http://localhost:8000/start");
    setSessionId(data.session_id);
    setStep({ type: "question", text: data.question, index: data.question_index });
    setHistory([]);
  };

  const sendAnswer = async (ans: 1 | 0 | -1) => {
    if (!sessionId || !step || step.type !== "question") return;

    const { data } = await axios.post("http://localhost:8000/answer", {
      session_id: sessionId,
      answer: ans,
    });

    setHistory((h) => [
      ...h,
      `${step.text} → ${ans === 1 ? "예" : ans === 0 ? "아니오" : "모르겠습니다"}`,
    ]);

    if (data.guess) {
      setStep({ type: "guess", name: data.guess_name });
    } else {
      setStep({ type: "question", text: data.question, index: data.question_index });
    }
  };

  /* ---------- UI ---------- */
  return (
    <main className="container-sm py-5">
      {/* --- 타이틀 / Hero --- */}
      {!step && (
        <section className="p-5 mb-4 bg-light rounded-3 text-center shadow-sm">
          <h1 className="display-5 fw-bold mb-3">
            <i className="bi bi-patch-question-fill me-2" />
            스무 고개로 동물 맞히기
          </h1>
          <p className="lead">
            예/아니오 질문에 답해&nbsp;보세요. 제가&nbsp;20 번 안에 여러분이
            생각한 동물을 맞혀볼게요!
          </p>
          <button className="btn btn-primary btn-lg px-5 mt-3" onClick={start}>
            게임 시작
          </button>
        </section>
      )}

      {/* --- 본 게임 카드 --- */}
      {step && (
        <div className="card mx-auto shadow" style={{ maxWidth: 540 }}>
          <div className="card-body text-center">
            {/* 진행 바 */}
            <div className="progress mb-4" style={{ height: 6 }}>
              <div
                className="progress-bar bg-success"
                role="progressbar"
                style={{ width: `${(history.length / 20) * 100}%` }} // 20회는 UX 표준치
                aria-valuenow={history.length}
                aria-valuemin={0}
                aria-valuemax={20}
              />
            </div>

            {/* 1) 질문 단계 */}
            {step.type === "question" && (
              <>
                <h2 className="h4 mb-4">{step.text}</h2>
                <div className="d-flex justify-content-center gap-2">
                  <button className="btn btn-success px-4" onClick={() => sendAnswer(1)}>
                    <i className="bi bi-check-circle me-1" />
                    예
                  </button>
                  <button className="btn btn-danger px-4" onClick={() => sendAnswer(0)}>
                    <i className="bi bi-x-circle me-1" />
                    아니오
                  </button>
                  <button
                    className="btn btn-secondary px-4"
                    onClick={() => sendAnswer(-1)}
                  >
                    <i className="bi bi-question-circle me-1" />
                    모르겠음
                  </button>
                </div>
              </>
            )}

            {/* 2) 추리 단계 */}
            {step.type === "guess" && (
              <>
                <h2 className="h4 mb-3">
                  <i className="bi bi-stars me-1 text-warning" />
                  제가 생각한 동물은…
                </h2>
                <p className="display-6 fw-semibold mb-4">{step.name}</p>
                <button className="btn btn-outline-primary" onClick={start}>
                  <i className="bi bi-arrow-repeat me-1" />
                  다시 하기
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* --- 질문/답 기록(Accordion) --- */}
      {history.length > 0 && (
        <div className="accordion mx-auto mt-4" style={{ maxWidth: 540 }} id="historyAcc">
          <div className="accordion-item">
            <h2 className="accordion-header" id="histHeading">
              <button
                className="accordion-button collapsed"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target="#histCollapse"
                aria-expanded="false"
                aria-controls="histCollapse"
              >
                <i className="bi bi-journal-text me-2" />
                질문 / 답 기록 ({history.length})
              </button>
            </h2>
            <div
              id="histCollapse"
              className="accordion-collapse collapse"
              aria-labelledby="histHeading"
              data-bs-parent="#historyAcc"
            >
              <ul className="list-group list-group-flush small">
                {history.map((h, i) => (
                  <li className="list-group-item" key={i}>
                    {h}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
