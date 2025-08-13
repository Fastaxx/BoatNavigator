from flask import Flask, request, jsonify, send_from_directory
import requests, math, datetime as dt
import numpy as np
import datetime as dt
from typing import Optional
import io
import datetime as dt
from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path="/static", static_folder="static")

# ---------- util géo ----------
R_EARTH = 6371000.0
deg2rad = np.deg2rad
rad2deg = np.rad2deg


def parse_iso_z(s: str) -> dt.datetime:
    """Parse ISO 8601, accepte 'Z' (UTC) et millisecondes. Renvoie tz-aware."""
    s2 = s.strip()
    if s2.endswith("Z"):
        s2 = s2[:-1] + "+00:00"
    # fromisoformat gère les fractions de secondes et les offsets (+HH:MM)
    return dt.datetime.fromisoformat(s2)


def haversine(p1, p2):
    lat1, lon1 = map(deg2rad, p1)
    lat2, lon2 = map(deg2rad, p2)
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R_EARTH*np.arcsin(np.sqrt(a))  # meters

def bearing(p1, p2):
    lat1, lon1 = map(deg2rad, p1)
    lat2, lon2 = map(deg2rad, p2)
    y = np.sin(lon2-lon1)*np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    brg = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return brg  # degrees

def angle_diff(a, b):
    d = (a - b + 540) % 360 - 180
    return abs(d)  # degrees

# ---------- Open-Meteo Marine ----------
def marine_vars():
    return {
      "hourly": [
        "wind_speed_10m","wind_direction_10m",
        "wave_height","wave_direction"
      ],
      "timezone": "auto"
    }

def hourly_pick_from(json_obj, keys, t_iso):
    ts = json_obj["hourly"]["time"]              # ex "2025-08-13T15:00"
    t_target = parse_iso_z(t_iso).replace(tzinfo=None)  # local naive
    idx = min(range(len(ts)), key=lambda i: abs(dt.datetime.fromisoformat(ts[i]) - t_target))
    out = {}
    for k in keys:
        out[k] = json_obj["hourly"][k][idx]
    return out

def hourly_interp_from(json_obj, keys_linear, keys_circular, t_iso):
    """Interpole linéairement (ou circulairement) à l'instant t_iso."""
    # t_iso peut être tz-aware; les times Open-Meteo avec timezone=auto sont naïfs locaux
    t_target = parse_iso_z(t_iso).replace(tzinfo=None)
    ts = [dt.datetime.fromisoformat(x) for x in json_obj["hourly"]["time"]]
    # bornes
    if t_target <= ts[0]:
        i0, i1, w = 0, 0, 0.0
    elif t_target >= ts[-1]:
        i0, i1, w = len(ts)-1, len(ts)-1, 0.0
    else:
        # trouve l'intervalle [i0, i1] tel que ts[i0] <= t < ts[i1]
        for i in range(len(ts)-1):
            if ts[i] <= t_target <= ts[i+1]:
                i0, i1 = i, i+1
                dt_tot = (ts[i1]-ts[i0]).total_seconds()
                w = 0.0 if dt_tot==0 else (t_target - ts[i0]).total_seconds()/dt_tot
                break

    out = {}
    # linéaire
    for k in keys_linear:
        v0 = float(json_obj["hourly"][k][i0])
        v1 = float(json_obj["hourly"][k][i1])
        out[k] = v0*(1-w) + v1*w

    # circulaire (degrés "from")
    for k in keys_circular:
        a0 = math.radians(float(json_obj["hourly"][k][i0]))
        a1 = math.radians(float(json_obj["hourly"][k][i1]))
        # interpole les composantes de la direction "to" pour éviter le wrap 0/360
        # (dir_to = dir_from + 180)
        to0 = (math.degrees(a0)+180.0) % 360.0
        to1 = (math.degrees(a1)+180.0) % 360.0
        u0, v0 = math.cos(math.radians(to0)), math.sin(math.radians(to0))
        u1, v1 = math.cos(math.radians(to1)), math.sin(math.radians(to1))
        u = u0*(1-w) + u1*w
        v = v0*(1-w) + v1*w
        to = (math.degrees(math.atan2(v, u)) + 360.0) % 360.0
        fr = (to + 180.0) % 360.0
        out[k] = fr
    return out



def fetch_wind(lat, lon, start_iso):
    t0 = parse_iso_z(start_iso)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": t0.date().isoformat(),
        "end_date": (t0 + dt.timedelta(hours=2)).date().isoformat(),  # couvre h+1 même si minuit
        "hourly": "wind_speed_10m,wind_direction_10m",
        "windspeed_unit": "kn",
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=20); r.raise_for_status()
    return r.json()

def fetch_waves(lat, lon, start_iso):
    t0 = parse_iso_z(start_iso)
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": t0.date().isoformat(),
        "end_date": (t0 + dt.timedelta(hours=2)).date().isoformat(),
        "hourly": "wave_height,wave_direction",
        "timezone": "auto",
        "cell_selection": "sea"
    }
    r = requests.get(url, params=params, timeout=20); r.raise_for_status()
    return r.json()



def hourly_pick(fx, t_iso):
    ts = fx["hourly"]["time"]  # e.g. "2025-08-13T15:00"
    # cible convertie depuis ISO possiblement 'Z' -> tz-aware, puis rendue naïve locale
    t_target_local_naive = parse_iso_z(t_iso).replace(tzinfo=None)
    idx = min(
        range(len(ts)),
        key=lambda i: abs(dt.datetime.fromisoformat(ts[i]) - t_target_local_naive)
    )
    return {
        k: fx["hourly"][k][idx]
        for k in ["wind_speed_10m","wind_direction_10m","wave_height","wave_direction"]
    }

def get_met_at(mid_lat, mid_lon, t_iso, use_wind, use_waves):
    met = {}
    if use_wind:
        w = fetch_wind(mid_lat, mid_lon, t_iso)
        met |= hourly_pick_from(w, ["wind_speed_10m","wind_direction_10m"], t_iso)
    if use_waves:
        v = fetch_waves(mid_lat, mid_lon, t_iso)
        met |= hourly_pick_from(v, ["wave_height","wave_direction"], t_iso)
    return met

def segment_eval(p1, p2, cap, d_m, t0_iso, v0_kn, C0_lph, pars, use_wind, use_waves):
    """Retourne vitesse, dt, litres pour 1 segment, selon options vent/vagues."""
    mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
    met = {}
    if use_wind:
        w = fetch_wind(mid[0], mid[1], t0_iso)
        met |= hourly_interp_from(w, ["wind_speed_10m"], ["wind_direction_10m"], t0_iso)
    if use_waves:
        v = fetch_waves(mid[0], mid[1], t0_iso)
        met |= hourly_interp_from(v, ["wave_height"], ["wave_direction"], t0_iso)

    # Valeurs par défaut si non demandées
    U = float(met.get("wind_speed_10m", 0.0))
    Dwind = float(met.get("wind_direction_10m", cap))  # si absent, aligné => cos(0)=1 mais U=0
    Hs = float(met.get("wave_height", 0.0))
    Dwave = float(met.get("wave_direction", cap))

    theta_wind = angle_diff(Dwind, cap)
    theta_wave = angle_diff(Dwave, cap)

    Uref,Href = pars.get("Uref",10.0), pars.get("Href",1.0)
    alpha, beta = pars.get("alpha",0.08), pars.get("beta",0.10)
    gamma, delta = pars.get("gamma",0.15), pars.get("delta",0.10)

    U_par = U * math.cos(math.radians(theta_wind))  # kn

    # Pénalité de vitesse (si vent/vagues off, U=0 ou Hs=0 => pas d’effet)
    penal_v = 1.0 - alpha*(U_par/Uref) - beta*(Hs/Href)*(math.cos(math.radians(theta_wave))**2)
    v_kn = max(0.5, v0_kn * penal_v)

    d_nm = d_m / 1852.0
    dt_h = d_nm / v_kn

    # Surconsommation (headwind et Hs)
    mult_c = 1.0 + gamma*max(U_par,0)/Uref + delta*(Hs/Href)
    C_l = C0_lph * mult_c * dt_h

    return v_kn, dt_h, C_l, met


# ---------- modèle conso ----------
def segment_consumption(p1, p2, t0_iso, v0_kn, C0_lph, pars):
    d_m = haversine(p1, p2)
    cap = bearing(p1, p2)
    mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

    wind = fetch_wind(mid[0], mid[1], t0_iso)         # <-- NOUVEAU
    waves = fetch_waves(mid[0], mid[1], t0_iso)       # <-- NOUVEAU

    met_w = hourly_pick_from(wind,  ["wind_speed_10m","wind_direction_10m"], t0_iso)
    met_wv = hourly_pick_from(waves, ["wave_height","wave_direction"], t0_iso)
    # fusion vue "météo"
    met = {**met_w, **met_wv}

    U = float(met["wind_speed_10m"])          # déjà en kn
    Dwind = float(met["wind_direction_10m"])  # deg (from)
    Hs = float(met["wave_height"])            # m
    Dwave = float(met["wave_direction"])      # deg (from)

    theta_wind = angle_diff(Dwind, cap)
    theta_wave = angle_diff(Dwave, cap)

    Uref,Href = pars.get("Uref",10.0), pars.get("Href",1.0)
    alpha, beta = pars.get("alpha",0.08), pars.get("beta",0.10)
    gamma, delta = pars.get("gamma",0.15), pars.get("delta",0.10)

    U_par = U * math.cos(math.radians(theta_wind))  # + headwind, - tailwind
    penal_v = 1.0 - alpha*(U_par/Uref) - beta*(Hs/Href)*(math.cos(math.radians(theta_wave))**2)
    v_kn = max(0.5, v0_kn * penal_v)

    d_nm = d_m / 1852.0
    dt_h = d_nm / v_kn
    mult_c = 1.0 + gamma*max(U_par,0)/Uref + delta*(Hs/Href)
    C_l = C0_lph * mult_c * dt_h

    return {
        "distance_m": d_m, "bearing_deg": cap,
        "eta_h": dt_h, "speed_kn": v_kn, "liters": C_l,
        "met": met
    }

# ---------- export ----------

def summarize_route(pts_latlon, start_iso, v0_kn, C0_lph, pars, use_wind, use_waves):
    """Calcule les segments + totaux (dist, temps, conso) avec ETA par segment."""
    segs = []
    tcur = parse_iso_z(start_iso)
    total_m = total_h = total_L = 0.0
    for i in range(len(pts_latlon)-1):
        p1, p2 = pts_latlon[i], pts_latlon[i+1]
        d_m = haversine(p1, p2)
        cap = bearing(p1, p2)
        v_kn, dt_h, C_l, _met = segment_eval(p1, p2, cap, d_m, tcur.isoformat(), v0_kn, C0_lph, pars, use_wind, use_waves)
        seg = {
            "i": i+1,
            "from": p1, "to": p2,
            "bearing": cap,
            "dist_nm": d_m/1852.0,
            "speed_kn": v_kn,
            "dt_h": dt_h,
            "t_start": tcur,
            "t_end": tcur + dt.timedelta(hours=dt_h),
            "liters": C_l
        }
        segs.append(seg)
        tcur = seg["t_end"]
        total_m += d_m
        total_h += dt_h
        total_L += C_l
    totals = {
        "distance_km": total_m/1000.0,
        "distance_nm": total_m/1852.0,
        "time_h": total_h,
        "liters": total_L,
        "eta_arrival": tcur
    }
    return segs, totals

def _plot_route_png(pts_latlon):
    """Génère un PNG (bytes) du tracé lat/lon (sans fond de carte)."""
    lats = [p[0] for p in pts_latlon]
    lons = [p[1] for p in pts_latlon]
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=150)
    ax.plot(lons, lats, marker="o")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Tracé (lat/lon)")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def _hms_from_hours(h):
    sec = int(round(h*3600))
    hh = sec//3600; mm=(sec%3600)//60; ss=sec%60
    return f"{hh}h{mm:02d}m"

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/consumption", methods=["POST"])
def api_consumption():
    data = request.get_json()
    route = data["route"]["coordinates"]               # [lng,lat]
    pts = [(lat, lon) for lon, lat in route]
    start_iso = data.get("start_iso")
    v0_kn = float(data.get("base_speed_kn", 8.0))
    C0_lph = float(data.get("lph", 5.0))
    pars = data.get("model", {})
    mode = data.get("mode", "corrected")               # "baseline" | "corrected" | "both"
    use_wind = bool(data.get("use_wind", True))
    use_waves = bool(data.get("use_waves", True))

    # Pré-calc distances & caps
    seg_geom = []
    for i in range(len(pts)-1):
        d_m = haversine(pts[i], pts[i+1])
        cap = bearing(pts[i], pts[i+1])
        seg_geom.append((d_m, cap))

    def compute_scenario(enable_wind, enable_waves):
        segs = []
        tcur = parse_iso_z(start_iso)
        for i in range(len(pts)-1):
            d_m, cap = seg_geom[i]
            v_kn, dt_h, C_l, met = segment_eval(
                pts[i], pts[i+1], cap, d_m, tcur.isoformat(),
                v0_kn, C0_lph, pars, enable_wind, enable_waves
            )
            segs.append({
                "distance_m": d_m, "bearing_deg": cap,
                "eta_h": dt_h, "speed_kn": v_kn, "liters": C_l,
                "met": met
            })
            tcur += dt.timedelta(hours=dt_h)
        return {
            "distance_km": sum(s["distance_m"] for s in segs)/1000.0,
            "time_h": sum(s["eta_h"] for s in segs),
            "liters": sum(s["liters"] for s in segs),
            "segments": segs
        }

    if mode == "baseline":
        base = compute_scenario(False, False)
        return jsonify({"baseline": base})
    elif mode == "both":
        base = compute_scenario(False, False)
        corr = compute_scenario(use_wind, use_waves)
        # Diff pratique
        diff = {
            "time_h": corr["time_h"] - base["time_h"],
            "liters": corr["liters"] - base["liters"]
        }
        return jsonify({"baseline": base, "corrected": corr, "delta": diff})
    else:  # "corrected"
        corr = compute_scenario(use_wind, use_waves)
        return jsonify({"corrected": corr})

@app.route("/export/gpx", methods=["POST"])
def export_gpx():
    data = request.get_json()
    route = data["route"]["coordinates"]  # [lng,lat]
    pts = [(lat, lon) for lon, lat in route]
    start_iso = data.get("start_iso")
    v0_kn = float(data.get("base_speed_kn", 8.0))
    C0_lph = float(data.get("lph", 5.0))
    pars = data.get("model", {})
    use_wind = bool(data.get("use_wind", True))
    use_waves = bool(data.get("use_waves", True))

    segs, totals = summarize_route(pts, start_iso, v0_kn, C0_lph, pars, use_wind, use_waves)

    # Horodatage des points (interp linéaire entre t_start et t_end des segments)
    times = [segs[0]["t_start"]]
    for s in segs:
        times.append(s["t_end"])
    # Construit GPX
    from xml.sax.saxutils import escape
    gpx = io.StringIO()
    gpx.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    gpx.write('<gpx version="1.1" creator="BoatNavigator" xmlns="http://www.topografix.com/GPX/1/1">\n')
    gpx.write(f'  <metadata><name>Navigation {escape(start_iso)}</name></metadata>\n')
    gpx.write('  <trk><name>Route</name><trkseg>\n')
    for (lat,lon), t in zip(pts, times):
        t_utc = t.astimezone(dt.timezone.utc)
        gpx.write(f'    <trkpt lat="{lat:.6f}" lon="{lon:.6f}"><time>{t_utc.isoformat().replace("+00:00","Z")}</time></trkpt>\n')
    gpx.write('  </trkseg></trk>\n')
    gpx.write('</gpx>\n')
    bio = io.BytesIO(gpx.getvalue().encode("utf-8"))
    bio.seek(0)
    return send_file(bio, mimetype="application/gpx+xml", as_attachment=True, download_name="navigation.gpx")

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    data = request.get_json()
    route = data["route"]["coordinates"]  # [lng,lat]
    pts = [(lat, lon) for lon, lat in route]
    start_iso = data.get("start_iso")
    v0_kn = float(data.get("base_speed_kn", 8.0))
    C0_lph = float(data.get("lph", 5.0))
    pars = data.get("model", {})
    use_wind = bool(data.get("use_wind", True))
    use_waves = bool(data.get("use_waves", True))

    segs, totals = summarize_route(pts, start_iso, v0_kn, C0_lph, pars, use_wind, use_waves)

    # Image du tracé
    img_buf = _plot_route_png(pts)

    # PDF
    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Résumé de navigation", styles["Title"])
    story += [title, Spacer(1, 8)]

    pinfo = [
        f"Départ : {parse_iso_z(start_iso).strftime('%Y-%m-%d %H:%M')}",
        f"Vitesse de base : {v0_kn:.1f} kn",
        f"Conso moyenne : {C0_lph:.1f} L/h",
        f"Options : vent={'ON' if use_wind else 'OFF'} ; vagues={'ON' if use_waves else 'OFF'}"
    ]
    for line in pinfo:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 8))

    # Carte (schéma lat/lon)
    story.append(Image(img_buf, width=400, height=400))
    story.append(Spacer(1, 8))

    # Tableau segments
    data_tbl = [["#", "De (lat,lon)", "À (lat,lon)", "Cap (°)", "Dist (NM)", "V (kn)", "Δt", "Début", "Fin", "L (L)"]]
    for s in segs:
        row = [
            s["i"],
            f"{s['from'][0]:.5f}, {s['from'][1]:.5f}",
            f"{s['to'][0]:.5f}, {s['to'][1]:.5f}",
            f"{s['bearing']:.0f}",
            f"{s['dist_nm']:.2f}",
            f"{s['speed_kn']:.1f}",
            _hms_from_hours(s["dt_h"]),
            s["t_start"].strftime("%H:%M"),
            s["t_end"].strftime("%H:%M"),
            f"{s['liters']:.2f}",
        ]
        data_tbl.append(row)

    tbl = Table(data_tbl, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
        ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("ALIGN",(3,1),(-1,-1), "CENTER"),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE")
    ]))
    story += [tbl, Spacer(1, 10)]

    # Totaux
    tot_p = Paragraph(
        f"<b>Totaux</b> — Distance : {totals['distance_nm']:.2f} NM "
        f"({totals['distance_km']:.1f} km) ; Temps : {_hms_from_hours(totals['time_h'])} ; "
        f"Carburant : {totals['liters']:.2f} L ; Arrivée prévue : {totals['eta_arrival'].strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    )
    story.append(tot_p)

    doc.build(story)
    pdf_buf.seek(0)
    return send_file(pdf_buf, mimetype="application/pdf", as_attachment=True, download_name="navigation.pdf")

if __name__ == "__main__":
    app.run(debug=True)
