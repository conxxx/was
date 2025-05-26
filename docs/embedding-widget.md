# Embedding Your Chatbot Widget

Follow these steps to embed the chatbot widget onto your website.

## 1. Find Your Chatbot ID

Your unique Chatbot ID is required to link the widget to your specific chatbot configuration.

1.  **Log in** to your chatbot platform account.
2.  Navigate to the **Dashboard** or **My Chatbots** section.
3.  Locate the chatbot you want to embed.
4.  Go to the chatbot's **Settings** or **Configuration** page.
5.  Look for a section labeled **Embed**, **Integration**, or **API Key**.
6.  You should find your **Chatbot ID** (it might also be called API Key or Widget ID) displayed clearly. Copy this ID.

*Example:* Your Chatbot ID might look something like `ch_a1b2c3d4e5f6g7h8`.

## 2. Add the Code Snippet to Your Website

You need to add a small piece of HTML and JavaScript code to your website's source code.

1.  **Edit the HTML** file of the page(s) where you want the chatbot widget to appear. This is often `index.html` or a template file in your website's theme.
2.  **Paste the following code snippet** just before the closing `</body>` tag:

```html
<!-- Chatbot Widget Start -->
<script>
  window.chatbotConfig = {
    chatbotId: 'YOUR_CHATBOT_ID' // <-- Replace with your actual Chatbot ID
  };
</script>
<script src="https://your-platform.com/cdn/widget.js" defer></script>
<!-- Chatbot Widget End -->
```

3.  **Replace `'YOUR_CHATBOT_ID'`** in the snippet with the actual Chatbot ID you copied in Step 1.
4.  **Save** the changes to your HTML file.

## 3. How the Widget Script Works

*   The `widget.js` script contains the logic for the chatbot widget interface.
*   This script is typically **hosted by the chatbot platform** on a Content Delivery Network (CDN), as shown in the example snippet (`https://your-platform.com/cdn/widget.js`). You usually don't need to download or host this file yourself. The platform ensures it's kept up-to-date.
*   The `defer` attribute ensures the script loads without blocking your website's rendering.
*   Once loaded, the script uses the `chatbotId` you provided to fetch your chatbot's specific settings and connect to the backend.

After completing these steps and publishing your updated website code, the chatbot widget should appear on the designated pages.