'use client'

import { useState } from 'react'
import axios from 'axios'

const questions = [
  "날 수 있나요?", "물을 좋아하나요?", "네 발로 걷나요?", "전기를 사용할 수 있나요?", "초식동물인가요?",
  "야행성인가요?", "사막에서 살 수 있나요?", "독이 있나요?", "인간보다 똑똑한가요?", "알을 낳나요?",
  "털이 있나요?", "차가운 곳에서도 살 수 있나요?", "집에서 키우기 적합한가요?", "바퀴가 있나요?",
  "이동할 수 있나요?", "식물을 먹나요?", "소리를 낼 수 있나요?", "크기가 사람보다 큰가요?",
  "두뇌가 있나요?", "위험한가요?"
]

export default function Home() {
  const [answers, setAnswers] = useState([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [result, setResult] = useState("")

  const handleAnswer = async (value) => {
    const updated = [...answers, value]
    setAnswers(updated)

    try {
      const res = await axios.post('http://localhost:8000/predict', {
        answers: updated
      })

      if (res.data.result) {
        setResult(res.data.result)
      } else if (currentIndex + 1 < questions.length) {
        setCurrentIndex(currentIndex + 1)
      } else {
        setResult("정확한 예측이 어려워요 😢")
      }
    } catch (err) {
      setResult("서버 오류 😢")
    }
  }

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>스무고개 AI</h1>

      {result ? (
        <h2>당신이 생각한 건... <b>{result}</b>인가요?</h2>
      ) : (
        <>
          <p><b>Q{currentIndex + 1}.</b> {questions[currentIndex]}</p>
          <div style={{ display: "flex", gap: "1rem" }}>
            <button onClick={() => handleAnswer(0)}>예</button>
            <button onClick={() => handleAnswer(1)}>아니요</button>
            <button onClick={() => handleAnswer(2)}>모르겠음</button>
          </div>
        </>
      )}
    </div>
  )
}
