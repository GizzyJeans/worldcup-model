"""塔羅球神諭 — Tarot Oracle for football matches. ⚜️🔮⚽

A wholly unserious counterpart to the Dixon-Coles model: it reads the 22 Major
Arcana instead of expected goals. The draw is DETERMINISTIC per fixture (seeded
by the team names), so each match has one fixed "fate" — re-running gives the
same cards. Three cards are pulled:

    主隊牌 (home)   the home side's energy
    客隊牌 (away)   the away side's energy
    結局牌 (verdict) the spread the cards favour

This predicts nothing. It is for fun. Bet with the model, laugh with the cards.

    python tarot_predict.py "South Africa" "Canada"
    python tarot_predict.py --r32          # read the whole Round of 32
"""

from __future__ import annotations

import argparse
import hashlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# Each Major Arcana: (name_en, name_zh, omen, tilt). `tilt` steers the verdict:
#   home / away  -> that side is favoured     draw -> stalemate energy
#   over / under -> goals / a cagey low score  upset -> the underdog's hour
#   chaos -> anything goes (penalties, red cards, late drama)
ARCANA = [
    ("The Fool", "愚者", "無畏的新生力量,初生之犢不畏虎。", "upset"),
    ("The Magician", "魔術師", "掌控全場、把握良機的一方終將點石成金。", "home"),
    ("The High Priestess", "女祭司", "深藏不露的防線,玄機盡在沉默之中。", "under"),
    ("The Empress", "皇后", "豐收與創造力—進球如泉湧。", "over"),
    ("The Emperor", "皇帝", "紀律與權威,強權者穩坐王座。", "home"),
    ("The Hierophant", "教皇", "傳統與經驗壓制新銳,老牌底蘊取勝。", "home"),
    ("The Lovers", "戀人", "勢均力敵、難分難捨,兩心相映成平局。", "draw"),
    ("The Chariot", "戰車", "氣勢如虹的一方策馬奔騰、強勢輾壓。", "away"),
    ("Strength", "力量", "以柔克剛,韌性與意志笑到最後。", "home"),
    ("The Hermit", "隱者", "謹慎保守、龜縮防守的低比分之夜。", "under"),
    ("Wheel of Fortune", "命運之輪", "天命無常—點球、烏龍、補時絕殺皆有可能。", "chaos"),
    ("Justice", "正義", "實力會被公正裁決,該贏的贏、強者勝。", "home"),
    ("The Hanged Man", "倒吊人", "僵局與犧牲,膠著難解的悶戰。", "draw"),
    ("Death", "死神", "舊秩序傾覆—大爆冷,黑馬踏破豪門。", "upset"),
    ("Temperance", "節制", "平衡與調和,雙方互有攻守、和氣收場。", "draw"),
    ("The Devil", "惡魔", "犯規、紅牌、暗黑誘惑—混亂中見勝負。", "chaos"),
    ("The Tower", "高塔", "崩塌與劇變,領先方城牆驟然倒下。", "upset"),
    ("The Star", "星星", "希望之光照耀客隊,遠來者得償所願。", "away"),
    ("The Moon", "月亮", "迷霧與假象,比分撲朔、爆冷暗湧。", "chaos"),
    ("The Sun", "太陽", "光明燦爛、進球盛宴—大比分之日。", "over"),
    ("Judgement", "審判", "覺醒與翻盤,落後者奮起反撲。", "upset"),
    ("The World", "世界", "圓滿與完成,熱門完美收官、強者加冕。", "home"),
]


def _draw(home: str, away: str) -> tuple[int, int, int]:
    """Deterministic three-card pull seeded by the fixture (no repeats)."""
    seed = hashlib.sha256(f"{home}|{away}".encode()).digest()
    idx, used = [], set()
    i = 0
    while len(idx) < 3:
        c = seed[i % len(seed)] ^ (i * 31 & 0xFF)
        card = c % len(ARCANA)
        if card not in used:
            used.add(card)
            idx.append(card)
        i += 1
    return idx[0], idx[1], idx[2]


def _verdict_line(home: str, away: str, tilt: str) -> str:
    return {
        "home": f"🏆 結局牌指向 **{home}**:主隊能量壓制全場。",
        "away": f"🏆 結局牌指向 **{away}**:客隊星象高照,爆冷或客勝。",
        "draw": f"🤝 結局牌示「平」:勢均力敵,90 分鐘難分勝負。",
        "over": f"⚽ 結局牌示「大分」:神諭預見進球盛宴(Over)。",
        "under": f"🔒 結局牌示「小分」:銅牆鐵壁,低比分之夜(Under)。",
        "upset": f"🐎 結局牌示「爆冷」:黑馬之日,弱者掀翻強權。",
        "chaos": f"🎲 結局牌示「混沌」:點球、紅牌、補時絕殺—一切皆有可能。",
    }[tilt]


def reading(home: str, away: str) -> str:
    h, a, v = _draw(home, away)
    hc, ac, vc = ARCANA[h], ARCANA[a], ARCANA[v]
    L = []
    L.append(f"  ┌─ 🔮 塔羅球神諭 ─ {home} vs {away} ─┐")
    L.append(f"   主隊牌  {hc[1]} ({hc[0]}) — {hc[2]}")
    L.append(f"   客隊牌  {ac[1]} ({ac[0]}) — {ac[2]}")
    L.append(f"   結局牌  {vc[1]} ({vc[0]}) — {vc[2]}")
    L.append(f"   {_verdict_line(home, away, vc[3])}")
    L.append(f"  └─ 命運已定,僅供一笑。實戰請看模型。 ─┘")
    return "\n".join(L)


# The actual 2026 Round of 32 (matches 73-88).
R32 = [
    ("South Africa", "Canada"), ("Germany", "Paraguay"),
    ("Netherlands", "Morocco"), ("Brazil", "Japan"),
    ("France", "Sweden"), ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"), ("England", "DR Congo"),
    ("United States", "Bosnia and Herzegovina"), ("Belgium", "Senegal"),
    ("Portugal", "Croatia"), ("Spain", "Austria"),
    ("Switzerland", "Algeria"), ("Argentina", "Cape Verde"),
    ("Colombia", "Ghana"), ("Australia", "Egypt"),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("home", nargs="?", help="home team")
    ap.add_argument("away", nargs="?", help="away team")
    ap.add_argument("--r32", action="store_true", help="read the whole Round of 32")
    args = ap.parse_args()

    print("\n   ✦ ⚜  塔羅球神諭  TAROT ORACLE  ⚜ ✦\n")
    if args.r32:
        for h, a in R32:
            print(reading(h, a) + "\n")
    elif args.home and args.away:
        print(reading(args.home, args.away) + "\n")
    else:
        ap.error("give a fixture: tarot_predict.py HOME AWAY   (or --r32)")


if __name__ == "__main__":
    main()
