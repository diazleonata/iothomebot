import asyncio
import subprocess
import os
import re
from PIL import Image, ImageChops, ImageStat
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))  # fallback 0 if unset
CAPTURE_FOLDER = "../storage/dcim/motionDetectCapture/"
MOTION_CHECK_INTERVAL = 4  # seconds
DIFFERENCE_THRESHOLD = 5  # adjustable threshold for mean pixel difference
MAX_DIFF_DIM = 320  # Downscale image before comparison

motion_enabled = False
torch_enabled = False


async def run_cmd_async(cmd: str) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip()
    except Exception as e:
        return f"[EXCEPTION] {e}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        ["/torch_toggle", "/motion_toggle"],
        ["/music"],
        ["/previous", "/playpause", "/next"],
        ["/run uptime"],
    ]
    await update.message.reply_text(
        "🔧 Choose an action:",
        reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True),
    )


async def run(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return await update.message.reply_text("🚫 Not authorized.")
    if not context.args:
        return await update.message.reply_text("⚠️ Usage: /run <command>")
    cmd = " ".join(context.args)
    output = await run_cmd_async(cmd)
    await update.message.reply_text(
        f"📥 Command:\n`{cmd}`\n📤 Output:\n```\n{output}\n```", parse_mode="Markdown"
    )


async def torch_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global torch_enabled
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return await update.message.reply_text("🚫 Not authorized.")
    torch_enabled = not torch_enabled
    action = "ON" if torch_enabled else "OFF"
    output = await run_cmd_async(f'am broadcast -a "BOT_FLASHLIGHT_{action}"')
    await update.message.reply_text(
        f"💡 Flashlight {action.lower()}ed\n```\n{output}\n```", parse_mode="Markdown"
    )


async def motion_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global motion_enabled
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return await update.message.reply_text("🚫 Not authorized.")
    motion_enabled = not motion_enabled
    action = "ON" if motion_enabled else "OFF"
    result = await run_cmd_async(f'am broadcast -a "BOT_MOTIONDETECT_{action}"')
    await update.message.reply_text(
        f"Motion {action.lower()}ed!\n```\n{result}\n```", parse_mode="Markdown"
    )


async def music(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return await update.message.reply_text("🚫 Not authorized.")
    if not context.args:
        return await update.message.reply_text("⚠️ Usage: /music <song name>")

    query = " ".join(context.args)
    await update.message.reply_text(f"🔍 Searching for: {query}")

    try:
        cmd = f'yt-dlp "ytsearch1:{query}" --print id --quiet --no-warnings'
        raw_output = await run_cmd_async(cmd)

        # Extract only valid 11-character YouTube video IDs
        matches = re.findall(r"[a-zA-Z0-9_-]{11}", raw_output)
        if not matches:
            return await update.message.reply_text("❌ No valid video ID found.")

        vid = matches[0]
        video_url = f"https://music.youtube.com/watch?v={vid}"

        await run_cmd_async(
            f"am start -n com.metrolist.music/.MainActivity "
            f"-a android.intent.action.VIEW "
            f'-d "{video_url}"'
        )
        await update.message.reply_text(f"🎵 Playing: {video_url}")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def playpause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return await update.message.reply_text("🚫 Not authorized.")
    output = await run_cmd_async('am broadcast -a "BOT_MEDIACONTROL_PLAYPAUSE"')
    await update.message.reply_text(f"⏯️ Play/Pause toggled", parse_mode="Markdown")


async def next_track(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return await update.message.reply_text("🚫 Not authorized.")
    output = await run_cmd_async('am broadcast -a "BOT_MEDIACONTROL_NEXT"')
    await update.message.reply_text(f"⏭️ Skipped to next", parse_mode="Markdown")


async def previous_track(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        return await update.message.reply_text("🚫 Not authorized.")
    output = await run_cmd_async('am broadcast -a "BOT_MEDIACONTROL_PREVIOUS"')
    await update.message.reply_text(f"⏮️ Went to previous", parse_mode="Markdown")


def preprocess_image_for_diff(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    w, h = img.size
    max_side = max(w, h)
    if max_side > MAX_DIFF_DIM:
        scale = MAX_DIFF_DIM / max_side
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def detect_motion(img1_path: str, img2_path: str) -> bool:
    try:
        with Image.open(img1_path) as im1_raw, Image.open(img2_path) as im2_raw:
            img1 = preprocess_image_for_diff(im1_raw)
            img2 = preprocess_image_for_diff(im2_raw)
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.LANCZOS)
            diff = ImageChops.difference(img1, img2)
            stat = ImageStat.Stat(diff)
            mean_diff = stat.mean[0]
            print(
                f"[Motion Check] {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} -> mean diff {mean_diff:.2f}"
            )
            return mean_diff > DIFFERENCE_THRESHOLD
    except Exception as e:
        print(f"[Error Comparing Images] {e}")
        return False


async def monitor_motion():
    global torch_enabled

    last_mtime = 0.0
    last_image_path = None
    first_run = True

    try:
        entries = [
            entry
            for entry in os.scandir(CAPTURE_FOLDER)
            if entry.is_file()
            and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if entries:
            latest = max(entries, key=lambda e: e.stat().st_mtime)
            last_mtime = latest.stat().st_mtime
            last_image_path = os.path.join(CAPTURE_FOLDER, latest.name)
    except Exception as e:
        print(f"[monitor_motion init error] {e}")

    while True:
        try:
            entries = [
                entry
                for entry in os.scandir(CAPTURE_FOLDER)
                if entry.is_file()
                and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            new_entries = [e for e in entries if e.stat().st_mtime > last_mtime]

            if new_entries:
                new_entries.sort(key=lambda e: e.stat().st_mtime)
                for entry in new_entries:
                    new_path = os.path.join(CAPTURE_FOLDER, entry.name)
                    new_mtime = entry.stat().st_mtime

                    if last_image_path and motion_enabled:
                        if first_run:
                            print("⏳ Waiting before first motion check...")
                            await asyncio.sleep(4)
                            first_run = False

                        if new_mtime - last_mtime > 30:
                            print("⚠️ Skipping comparison due to large time gap.")
                        else:
                            if detect_motion(last_image_path, new_path):
                                print("🚨 Motion detected! Broadcasting flashlight ON.")
                                asyncio.create_task(
                                    run_cmd_async('am broadcast -a "BOT_FLASHLIGHT_ON"')
                                )
                                torch_enabled = True

                    last_image_path = new_path
                    last_mtime = new_mtime

        except Exception as e:
            print(f"[monitor_motion error] {e}")

        await asyncio.sleep(MOTION_CHECK_INTERVAL)


async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", run))
    app.add_handler(CommandHandler("torch_toggle", torch_toggle))
    app.add_handler(CommandHandler("motion_toggle", motion_toggle))
    app.add_handler(CommandHandler("music", music))
    app.add_handler(CommandHandler("playpause", playpause))
    app.add_handler(CommandHandler("next", next_track))
    app.add_handler(CommandHandler("previous", previous_track))

    print("🤖 Bot is running...")
    asyncio.create_task(monitor_motion())
    await app.run_polling()


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.run(main())
