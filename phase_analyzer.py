#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Фазовый анализатор диалогов v0.1

Идея:
- Вход: текстовый лог диалога (User / Assistant).
- Выход: JSON с:
    * пометками шагов (A,B,C,T),
    * эпизодами (пути между "аттракторами"),
    * оценкой первого "ошибочного уровня" системы.

Зависимости: только стандартная библиотека.
Можно расширять, подменяя простую bag-of-words модель на настоящие эмбеддинги.
"""

import sys
import json
import math
import re
from collections import defaultdict, namedtuple
from statistics import median, quantiles

Message = namedtuple("Message", ["index", "role", "text"])
StepMetrics = namedtuple(
    "StepMetrics",
    ["step_index", "user_idx", "assistant_idx",
     "divergence", "error_score", "jump", "T"]
)

# ---------------------------
# 1. Парсер диалога
# ---------------------------

def parse_chat_simple(path):
    """
    Очень простой парсер.
    Формат:
        User: текст
        Assistant: текст
    Последующие строки без префикса считаются продолжением предыдущего сообщения.
    """
    messages = []
    current_role = None
    current_text = []

    user_re = re.compile(r"^\s*(User|Юзер|Пользователь)\s*:\s*(.*)$", re.IGNORECASE)
    asst_re = re.compile(r"^\s*(Assistant|AI|ChatGPT)\s*:\s*(.*)$", re.IGNORECASE)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                # пустая строка — просто разделитель
                continue

            m_user = user_re.match(line)
            m_asst = asst_re.match(line)

            if m_user:
                # завершить предыдущее сообщение
                if current_role is not None and current_text:
                    idx = len(messages)
                    messages.append(
                        Message(index=idx, role=current_role, text="\n".join(current_text).strip())
                    )
                current_role = "user"
                current_text = [m_user.group(2)]

            elif m_asst:
                if current_role is not None and current_text:
                    idx = len(messages)
                    messages.append(
                        Message(index=idx, role=current_role, text="\n".join(current_text).strip())
                    )
                current_role = "assistant"
                current_text = [m_asst.group(2)]
            else:
                # продолжение предыдущего сообщения
                if current_role is None:
                    # строка до первого маркера — можно проигнорировать или считать системной
                    continue
                current_text.append(line)

    # финальное сообщение
    if current_role is not None and current_text:
        idx = len(messages)
        messages.append(
            Message(index=idx, role=current_role, text="\n".join(current_text).strip())
        )

    return messages


def build_steps(messages):
    """
    Строим пары (user -> assistant) как "шаги" диалога.
    """
    steps = []
    user_idx = None

    for msg in messages:
        if msg.role == "user":
            user_idx = msg.index
        elif msg.role == "assistant" and user_idx is not None:
            step_index = len(steps)
            steps.append((step_index, user_idx, msg.index))
            user_idx = None

    return steps


# ---------------------------
# 2. Примитивная "эмбеддинг"-модель
# ---------------------------

def tokenize(text):
    text = text.lower()
    # убираем всё, кроме букв и цифр
    text = re.sub(r"[^a-zа-яё0-9]+", " ", text, flags=re.IGNORECASE)
    tokens = [t for t in text.split() if t]
    return tokens


class BagOfWordsEmbedder:
    """
    Простейшая модель:
    - собирает словарь и document-frequency по всем сообщениям,
    - строит TF-IDF-вектора как dict{token: weight},
    - считает косинусное сходство.
    """

    def __init__(self, messages):
        self.df = defaultdict(int)
        self.N = len(messages)
        self._build_df(messages)

    def _build_df(self, messages):
        for msg in messages:
            tokens = set(tokenize(msg.text))
            for t in tokens:
                self.df[t] += 1

    def embed(self, text):
        tokens = tokenize(text)
        if not tokens:
            return {}
        tf = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        # нормировка TF
        length = float(len(tokens))
        for t in tf:
            tf[t] /= length
        # IDF и TF-IDF
        vec = {}
        for t, tf_val in tf.items():
            df = self.df.get(t, 1)
            idf = math.log((self.N + 1.0) / (df + 0.5))
            vec[t] = tf_val * idf
        return vec

    @staticmethod
    def cosine(v1, v2):
        if not v1 or not v2:
            return 0.0
        # скалярное
        dot = 0.0
        for k, v in v1.items():
            if k in v2:
                dot += v * v2[k]
        # нормы
        n1 = math.sqrt(sum(v * v for v in v1.values()))
        n2 = math.sqrt(sum(v * v for v in v2.values()))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)


# ---------------------------
# 3. Метрики A,B,C и T
# ---------------------------

def text_len_score(u_text, a_text):
    """
    Простая метрика по длине: сильно несоответствующая длина ответа = ошибка.
    """
    nu = len(u_text)
    na = len(a_text)
    if nu == 0 or na == 0:
        return 1.0
    ratio = na / nu
    # идеал — примерно тот же порядок: 0.5 .. 2.0
    if 0.5 <= ratio <= 2.0:
        return 0.0
    # чем сильнее отклонение, тем больше ошибка, максимум 1.0
    dev = abs(math.log(ratio, 2))  # в лог-шкале
    return min(1.0, dev / 4.0)


def lexical_overlap_score(u_text, a_text):
    """
    Чем меньше пересечение лексики, тем больше ошибка.
    (но это только один из факторов)
    """
    u_set = set(tokenize(u_text))
    a_set = set(tokenize(a_text))
    if not u_set or not a_set:
        return 1.0
    inter = len(u_set & a_set)
    union = len(u_set | a_set)
    jaccard = inter / union
    # ошибка = 1 - jaccard (но не меньше 0.0)
    return 1.0 - jaccard


def compute_step_metrics(messages, steps):
    """
    Считает для каждого шага:
    - divergence (A)
    - error_score (B)
    - jump (C)
    - T = w1*A + w2*B + w3*C
    """
    embedder = BagOfWordsEmbedder(messages)

    # Подготовим эмбеддинги для всех сообщений
    embeddings = [embedder.embed(m.text) for m in messages]

    step_metrics = []

    prev_assistant_vec = None

    # веса в общей фазовой метрике
    w1, w2, w3 = 0.5, 0.3, 0.2

    for (step_idx, u_idx, a_idx) in steps:
        m_u = messages[u_idx]
        m_a = messages[a_idx]

        u_vec = embeddings[u_idx]
        a_vec = embeddings[a_idx]

        # A: смысловое расхождение (1 - cosine)
        sim = embedder.cosine(u_vec, a_vec)
        divergence = 1.0 - sim  # 0..1

        # B: error_score ~ сочетание длины и лексического попадания
        length_err = text_len_score(m_u.text, m_a.text)
        lex_err = lexical_overlap_score(m_u.text, m_a.text)
        error_score = 0.5 * length_err + 0.5 * lex_err
        error_score = max(0.0, min(1.0, error_score))

        # C: jump = изменение между текущим ответом и предыдущим ответом
        if prev_assistant_vec is None:
            jump = 0.0
        else:
            sim_prev = embedder.cosine(prev_assistant_vec, a_vec)
            jump = 1.0 - sim_prev
        prev_assistant_vec = a_vec

        T = w1 * divergence + w2 * error_score + w3 * jump

        step_metrics.append(
            StepMetrics(
                step_index=step_idx,
                user_idx=u_idx,
                assistant_idx=a_idx,
                divergence=divergence,
                error_score=error_score,
                jump=jump,
                T=T,
            )
        )

    return step_metrics


# ---------------------------
# 4. Эпизоды, φ* и "уровни"
# ---------------------------

def detect_episodes(step_metrics):
    """
    Эпизоды = непрерывные участки, где T выше медианы.
    Это грубо, но даёт каркас.
    """
    if not step_metrics:
        return []

    T_values = [s.T for s in step_metrics]
    med_T = median(T_values)

    episodes = []
    i = 0
    n = len(step_metrics)

    while i < n:
        if step_metrics[i].T > med_T:
            start = i
            while i + 1 < n and step_metrics[i + 1].T > med_T:
                i += 1
            end = i
            episodes.append((start, end))
        i += 1

    return episodes


def classify_episode_level(ep_metrics):
    """
    Условная классификация "уровня сбоя системы" для эпизода:
    0 — нормальный режим
    1 — локальные сбои
    2 — системный сбой (критический)
    """
    avg_div = ep_metrics["avg_divergence"]
    avg_err = ep_metrics["avg_error"]
    avg_jump = ep_metrics["avg_jump"]

    # простая эвристика
    if avg_err < 0.3 and avg_div < 0.3 and avg_jump < 0.3:
        return 0
    if avg_err < 0.6 and avg_div < 0.6:
        return 1
    return 2


def build_episode_report(messages, step_metrics, episodes):
    """
    Строит структуру эпизодов, считает:
    - L (сумма T),
    - φ* (кумулятивный шаг),
    - уровень.
    """
    report_episodes = []
    phi_prev = 1.0  # начальное φ*

    for ep_id, (start, end) in enumerate(episodes, start=1):
        steps = step_metrics[start:end+1]

        L = sum(s.T for s in steps)
        avg_div = sum(s.divergence for s in steps) / len(steps)
        avg_err = sum(s.error_score for s in steps) / len(steps)
        avg_jump = sum(s.jump for s in steps) / len(steps)

        # простая модель φ*: кумулятивное наращивание
        phi_star = phi_prev + L
        phi_prev = phi_star

        ep_metrics = {
            "episode_id": ep_id,
            "start_step": steps[0].step_index,
            "end_step": steps[-1].step_index,
            "start_user_idx": steps[0].user_idx,
            "end_assistant_idx": steps[-1].assistant_idx,
            "num_steps": len(steps),
            "L": L,
            "avg_divergence": avg_div,
            "avg_error": avg_err,
            "avg_jump": avg_jump,
            "phi_star": phi_star,
        }

        level = classify_episode_level(ep_metrics)
        ep_metrics["system_level"] = level

        report_episodes.append(ep_metrics)

    return report_episodes


# ---------------------------
# 5. Основная функция: анализ и запись JSON
# ---------------------------

def analyze_dialog(input_path, output_path):
    messages = parse_chat_simple(input_path)
    if not messages:
        print("Нет сообщений в файле.", file=sys.stderr)
        return

    steps = build_steps(messages)
    if not steps:
        print("Не найдено пар (User → Assistant).", file=sys.stderr)
        return

    step_metrics = compute_step_metrics(messages, steps)
    episodes = detect_episodes(step_metrics)
    episode_report = build_episode_report(messages, step_metrics, episodes)

    # ищем первый эпизод с system_level >= 1
    first_problem_level = None
    first_problem_episode = None
    for ep in episode_report:
        if ep["system_level"] >= 1:
            first_problem_level = ep["system_level"]
            first_problem_episode = ep["episode_id"]
            break

    data = {
        "input_file": input_path,
        "num_messages": len(messages),
        "num_steps": len(step_metrics),
        "num_episodes": len(episode_report),
        "first_problem_level": first_problem_level,
        "first_problem_episode": first_problem_episode,
        "messages": [
            {
                "index": m.index,
                "role": m.role,
                "text": m.text,
            }
            for m in messages
        ],
        "steps": [
            {
                "step_index": s.step_index,
                "user_idx": s.user_idx,
                "assistant_idx": s.assistant_idx,
                "divergence": s.divergence,
                "error_score": s.error_score,
                "jump": s.jump,
                "T": s.T,
            }
            for s in step_metrics
        ],
        "episodes": episode_report,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Анализ завершён. JSON сохранён в {output_path}")
    if first_problem_level is not None:
        print(f"Первый 'ошибочный уровень' системы: {first_problem_level}, эпизод {first_problem_episode}")
    else:
        print("Серьёзных эпизодических сбоев не обнаружено (по текущим порогам).")


# ---------------------------
# 6. Точка входа
# ---------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python phase_analyzer.py dialog.txt report.json")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    analyze_dialog(in_path, out_path)
