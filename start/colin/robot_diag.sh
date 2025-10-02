#sudo tee /usr/local/bin/robot_diag.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
# robot_diag.sh — robust 1 Hz collector (no strict mode), writes to /tmp/robot_diag_YYYYmmdd_HHMMSS
# Optional env: CONTROL_PC=10.20.0.2  IFACE=wlan0|wlan1  PORTS=":8000,:5600"  DURATION=300  INTERVAL=1

CONTROL_PC="${CONTROL_PC:-}"
IF="${IFACE:-}"
INTERVAL="${INTERVAL:-1}"
DURATION="${DURATION:-0}"

# --- helpers ---
ts_ms(){ date +%s%3N; }
ts_iso(){ date -Iseconds; }
hz_to_ghz(){ awk '{printf "%.2f", ($1)/1e9}'; }
kb_to_gb(){ awk '{printf "%.2f", $1/1048576}'; }
has(){ command -v "$1" >/dev/null 2>&1; }

pick_iface(){
  # route to control pc if provided
  if [ -n "$CONTROL_PC" ]; then
    dev=$(ip route get "$CONTROL_PC" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
    [ -n "$dev" ] && echo "$dev" && return
  fi
  # AP with station
  if has iw; then
    for ifc in $(iw dev 2>/dev/null | awk '/Interface/{print $2}'); do
      if iw dev "$ifc" info 2>/dev/null | grep -q 'type AP' && iw dev "$ifc" station dump 2>/dev/null | grep -q '^Station'; then
        echo "$ifc"; return
      fi
    done
    # any UP wifi
    for ifc in $(iw dev 2>/dev/null | awk '/Interface/{print $2}'); do
      [ "$(cat /sys/class/net/$ifc/operstate 2>/dev/null)" = "up" ] && echo "$ifc" && return
    done
  fi
  # first UP iface
  ip -br link | awk '$2=="UP"{print $1; exit}'
}

OUTDIR="${OUTDIR:-/tmp/robot_diag_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"
ERR="$OUTDIR/errors.log"
echo "# $(ts_iso) starting robot_diag" > "$ERR"
echo "Output dir: $OUTDIR"

# choose iface
IF="${IF:-$(pick_iface)}"
if [ -z "$IF" ]; then
  echo "No active iface found" | tee -a "$ERR"
  exit 1
fi
echo "Using iface: $IF"

# tail kernel warnings (bg)
DMESG_LOG="$OUTDIR/dmesg_watch.log"
( sudo dmesg -w 2>>"$ERR" | egrep -i 'uvc|video[0-9]|usb|xHCI|bandwidth|urb|reset|disconnect|brcmf|wlan|under-volt|throttl' >> "$DMESG_LOG" ) &
DMESG_PID=$!
trap 'kill $DMESG_PID 2>/dev/null; echo "# $(ts_iso) stopped" >> "$ERR"' EXIT

# one-shot snapshots
SNAP="$OUTDIR/snapshot.txt"
{
  echo "# $(ts_iso) snapshots"
  echo "=== iw dev ==="; has iw && iw dev || echo "(no iw)"
  echo; echo "=== ip -br link ==="; ip -br link
  [ -n "$CONTROL_PC" ] && { echo; echo "=== ip route get $CONTROL_PC ==="; ip route get "$CONTROL_PC" || true; }
  echo; echo "=== lsusb -t ==="; lsusb -t 2>/dev/null || echo "(no lsusb -t)"
} > "$SNAP" 2>>"$ERR"

# camera info (if v4l2-ctl present)
if has v4l2-ctl; then
  CAMS="$OUTDIR/camera_formats.txt"
  for v in /dev/video*; do
    [ -e "$v" ] || continue
    {
      echo "=== $v ==="
      v4l2-ctl -d "$v" --get-fmt-video --get-parm
    } >> "$CAMS" 2>>"$ERR"
  done
fi

# CSV header
MAIN="$OUTDIR/main.csv"
echo "time_iso,epoch_ms,iface,tx_mbps,rx_mbps,tx_phy_mbps,rx_phy_mbps,rssi_dbm,inactive_ms,cpu_user,cpu_sys,tempC,arm_GHz,throttled_hex,mem_used_GB,mem_free_GB,channel,busy_pct,rtt_ms,jitter_ms" > "$MAIN"

# baselines
TX0=$(cat /sys/class/net/$IF/statistics/tx_bytes 2>/dev/null || echo 0)
RX0=$(cat /sys/class/net/$IF/statistics/rx_bytes 2>/dev/null || echo 0)
read _ a b c d _ _ _ _ _ < /proc/stat
PT=$((a+b+c+d)); PU=$a; PS=$c

JITTER="0.00"; PREV_RTT=""; iter=0; started=$(date +%s)

# progress ticker
echo "Writing: $MAIN  (progress line every ~5s)"
while :; do
  now_iso=$(ts_iso); now_ms=$(ts_ms)

  # cpu %
  read _ a b c d _ _ _ _ _ < /proc/stat
  T=$((a+b+c+d)); DT=$((T-PT)); DU=$((a-PU)); DS=$((c-PS))
  CPUU=$(( DT>0 ? (100*DU/DT) : 0 )); CPUS=$(( DT>0 ? (100*DS/DT) : 0 ))
  PT=$T; PU=$a; PS=$c

  # temp/freq/throttle (tolerate missing vcgencmd)
  if has vcgencmd; then
    TEMP=$(vcgencmd measure_temp 2>>"$ERR" | tr -dc '0-9.')
    THR=$(vcgencmd get_throttled 2>>"$ERR" | awk -F= '{print $2}')
    FREQ=$(vcgencmd measure_clock arm 2>>"$ERR" | awk -F= '{print $2}' | hz_to_ghz)
  else
    TEMP=$(awk '{printf "%.1f",$1/1000}' </sys/class/thermal/thermal_zone0/temp 2>/dev/null)
    THR="0x0"
    FREQ="$(awk '/cpu MHz/{printf "%.2f",$4/1000;exit}' /proc/cpuinfo)"
  fi
  [ -z "$TEMP" ] && TEMP=0; [ -z "$THR" ] && THR=0x0; [ -z "$FREQ" ] && FREQ=0

  # mem used/free
  MEMT=$(awk '/MemTotal/{print $2}' /proc/meminfo); MEMA=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  MEMU=$((MEMT-MEMA)); MEMU_G=$(echo "$MEMU" | kb_to_gb); MEMF_G=$(echo "$MEMA" | kb_to_gb)

  # iface bytes → Mb/s
  CT=$(cat /sys/class/net/$IF/statistics/tx_bytes 2>/dev/null || echo 0)
  CR=$(cat /sys/class/net/$IF/statistics/rx_bytes 2>/dev/null || echo 0)
  TX=$(( (CT-TX0)*8/1000000 )); RX=$(( (CR-RX0)*8/1000000 ))
  TX0=$CT; RX0=$CR

  # wifi stats (AP vs STA)
  TXP=""; RXP=""; RSSI=""; INACT=""
  if has iw && iw dev "$IF" info 2>/dev/null | grep -q 'type AP'; then
    ST="$(iw dev "$IF" station dump 2>/dev/null)"
    TXP=$(awk '/tx bitrate:/{print $3; exit}' <<<"$ST")
    RXP=$(awk '/rx bitrate:/{print $3; exit}' <<<"$ST")
    RSSI=$(awk '/signal:/{print $2; exit}' <<<"$ST")
    INACT=$(awk '/inactive time/{print $3; exit}' <<<"$ST")
  elif has iw; then
    LINK="$(iw dev "$IF" link 2>/dev/null)"
    TXP=$(awk '/tx bitrate:/{print $3}' <<<"$LINK")
    RXP=$(awk '/rx bitrate:/{print $3}' <<<"$LINK")
    RSSI=$(awk '/signal:/{print $2}' <<<"$LINK")
  fi

  # channel busy%
  CH=""; BUSY=""
  if has iw; then
    dump="$(iw dev "$IF" survey dump 2>/dev/null)"
    CH=$(awk '/channel/ {c=$2} /in use/ {iu=1} /active time/ {a=$3} END{if(a>0)print c; else print ""}' <<<"$dump")
    BUSY=$(awk '/busy time/ {b=$3} /active time/ {a=$3} END{if(a>0) printf "%d", (b*100)/a; else print ""}' <<<"$dump")
  fi

  # rtt/jitter
  RTT=""
  if [ -n "$CONTROL_PC" ]; then
    RTT=$(ping -n -c1 -W 1 "$CONTROL_PC" 2>/dev/null | awk -F'time=' '/time=/{print $2}' | awk '{print $1}')
    if [[ "$RTT" =~ ^[0-9.]+$ ]]; then
      if [ -n "$PREV_RTT" ]; then
        JITTER=$(awk -v j="$JITTER" -v r="$RTT" -v p="$PREV_RTT" 'BEGIN{d=r-p; if(d<0)d=-d; printf "%.2f", 0.9*j + 0.1*d}')
      else
        JITTER="0.00"
      fi
      PREV_RTT="$RTT"
    fi
  fi

  # write CSV
  echo "$now_iso,$now_ms,$IF,$TX,$RX,$TXP,$RXP,$RSSI,$INACT,$CPUU,$CPUS,$TEMP,$FREQ,$THR,$MEMU_G,$MEMF_G,$CH,$BUSY,$RTT,$JITTER" >> "$MAIN" 2>>"$ERR"

  # optional socket snapshot
  if [ -n "$PORTS" ]; then
    SS_LOG="$OUTDIR/ss_ports.log"
    {
      echo "### $(ts_iso)"
      ss -tinup
    } >> "$SS_LOG" 2>>"$ERR"
  fi

  # progress every ~5s
  iter=$((iter+1))
  if (( iter % 5 == 0 )); then
    echo "$(ts_iso) wrote $iter lines → $MAIN (IF=$IF TX=${TX}Mb/s RX=${RX}Mb/s)"
  fi

  # duration stop
  if [ "$DURATION" -gt 0 ] && [ $(( $(date +%s) - started )) -ge "$DURATION" ]; then
    break
  fi

  sleep "$INTERVAL"
done

echo "Done. Files in $OUTDIR"
EOF
sudo chmod +x /usr/local/bin/robot_diag.sh
