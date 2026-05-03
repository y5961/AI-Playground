import asyncio
from mcp.server.fastmcp import FastMCP
from playwright.async_api import Browser, Page, async_playwright
from typing import Optional

mcp = FastMCP("weather-Israel")

# Global state for browser and page
browser: Optional[Browser] = None
page: Optional[Page] = None
state_lock = asyncio.Lock()

FORECAST_URL = "https://www.weather2day.co.il/forecast"


async def ensure_browser_open() -> tuple[Browser, Page]:
    """Ensure browser is open and return browser and page objects."""
    global browser, page
    
    async with state_lock:
        if browser is None or page is None:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--ignore-certificate-errors",
                    "--disable-dev-shm-usage"
                ]
            )
            # Create context with SSL verification disabled
            context = await browser.new_context(
                ignore_https_errors=True
            )
            page = await context.new_page()
            # Set viewport for consistent rendering
            await page.set_viewport_size({"width": 1280, "height": 720})
            # Set user agent
            await page.context.set_extra_http_headers({"User-Agent": "Mozilla/5.0"})
        
        return browser, page


async def close_browser() -> None:
    """Close the browser."""
    global browser, page
    
    async with state_lock:
        if page is not None:
            await page.close()
            page = None
        if browser is not None:
            await browser.close()
            browser = None


@mcp.tool()
async def open_weather_forecast_israel() -> str:
    """Open the browser and navigate to the weather forecast page.
    
    This tool opens a Chromium browser and navigates to the Israeli weather 
    forecast website (weather2day.co.il).
    """
    try:
        _, page = await ensure_browser_open()
        
        # Try navigation with minimal timeout
        try:
            print(f"Opening {FORECAST_URL}...")
            # Use commit instead of load for faster response
            await asyncio.wait_for(
                page.goto(FORECAST_URL, wait_until="commit"),
                timeout=5.0
            )
            print("✅ Page commit loaded")
            return "✅ Browser opened and navigated to weather forecast page"
        except asyncio.TimeoutError:
            print("⚠️ Navigation timed out, but continuing with partial page")
            return "⚠️ Navigated with timeout - page may be partially loaded"
        except Exception as e:
            print(f"⚠️ Navigation error: {str(e)}")
            return f"⚠️ Navigation encountered an issue but continuing: {str(e)[:100]}"
        
    except Exception as e:
        error_msg = f"❌ Error opening browser: {str(e)}"
        print(error_msg)
        return error_msg


@mcp.tool()
async def enter_weather_forecast_city_israel(city: str) -> str:
    """Enter a city name in the search field.
    
    Args:
        city: City name in Hebrew or English (e.g., "תל אביב" or "Tel Aviv")
    """
    try:
        _, page = await ensure_browser_open()
        
        # Wait for search input to be available
        search_selectors = [
            "input[placeholder*='ערים']",
            "input[placeholder*='עיר']",
            "input[placeholder*='city']",
            "input[type='search']",
            "input[type='text']"
        ]
        
        search_input = None
        for selector in search_selectors:
            try:
                search_input = await page.query_selector(selector)
                if search_input:
                    break
            except:
                continue
        
        if not search_input:
            return f"❌ Could not find search input field"
        
        # Clear any existing text and enter the city
        await search_input.fill("")
        await search_input.fill(city)
        
        # Wait for dropdown to appear
        await page.wait_for_timeout(1500)
        
        return f"✅ Entered '{city}' in search field"
        
    except Exception as e:
        return f"❌ Error entering city: {str(e)}"


@mcp.tool()
async def select_weather_forecast_city_israel() -> str:
    """Select the first city from the dropdown list.
    
    This tool clicks on the first item in the dropdown list that appears
    after entering a city name.
    """
    try:
        _, page = await ensure_browser_open()
        
        # Wait for dropdown items to appear
        await page.wait_for_timeout(500)
        
        # Try different selectors for dropdown items
        dropdown_selectors = [
            ".dropdown-item:first-child",
            "[role='option']:first-of-type",
            ".option:first-child",
            "li:first-child",
            ".city-item:first-child",
            "div[class*='item']:first-child"
        ]
        
        clicked = False
        for selector in dropdown_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    await element.click()
                    clicked = True
                    break
            except:
                continue
        
        if not clicked:
            # Try using keyboard navigation as fallback
            await page.keyboard.press("ArrowDown")
            await page.keyboard.press("Enter")
        
        # Wait for page to load the forecast
        await page.wait_for_timeout(2000)
        
        return "✅ Selected city from dropdown"
        
    except Exception as e:
        return f"❌ Error selecting city: {str(e)}"


@mcp.tool()
async def extract_weather_forecast_israel() -> str:
    """Extract weather forecast data from the current page.
    
    This tool reads the weather information from the loaded page and
    returns a cleaned, readable format with temperature, conditions, and wind.
    """
    try:
        _, page = await ensure_browser_open()
        
        # Extract text content
        body_text = await page.evaluate("""
            () => {
                // Remove script and style elements
                const scripts = document.querySelectorAll('script, style');
                scripts.forEach(script => script.remove());
                
                // Get visible text
                let text = document.body.innerText;
                
                // Clean up extra whitespace
                text = text.replace(/\\n\\s*\\n/g, '\\n');
                
                return text;
            }
        """)
        
        # Try to extract specific weather data using common element patterns
        weather_data = await page.evaluate("""
            () => {
                const data = {};
                
                // Try to find city name
                const city_elements = document.querySelectorAll('h1, h2, [class*="city"], [class*="location"]');
                for (let elem of city_elements) {
                    const text = elem.innerText?.trim();
                    if (text && text.length > 0 && text.length < 50) {
                        data.city = text;
                        break;
                    }
                }
                
                // Try to find temperature
                const temp_elements = document.querySelectorAll('[class*="temp"], [class*="celsius"], [class*="degree"]');
                const temps = [];
                for (let elem of temp_elements) {
                    const text = elem.innerText?.trim();
                    if (text && /\\d+°/.test(text)) {
                        temps.push(text);
                    }
                }
                if (temps.length > 0) data.temperatures = temps.slice(0, 3);
                
                // Try to find conditions/description
                const desc_elements = document.querySelectorAll('[class*="condition"], [class*="weather"], [class*="forecast"], p');
                const descriptions = [];
                for (let elem of desc_elements) {
                    const text = elem.innerText?.trim();
                    if (text && text.length > 10 && text.length < 200 && !descriptions.includes(text)) {
                        descriptions.push(text);
                    }
                }
                if (descriptions.length > 0) data.conditions = descriptions.slice(0, 5);
                
                return data;
            }
        """)
        
        # Format the output
        result = "🌤️ **Weather Forecast Israel**\n\n"
        
        if weather_data.get('city'):
            result += f"📍 **Location:** {weather_data['city']}\n"
        
        if weather_data.get('temperatures'):
            result += f"🌡️ **Temperatures:** {', '.join(weather_data['temperatures'])}\n"
        
        if weather_data.get('conditions'):
            result += f"📝 **Conditions:**\n"
            for condition in weather_data['conditions']:
                result += f"  • {condition}\n"
        
        # If we couldn't extract specific data, return the full body text
        if not (weather_data.get('city') or weather_data.get('temperatures')):
            result += f"\n**Raw Page Content (First 1000 chars):**\n{body_text[:1000]}..."
        
        return result
        
    except Exception as e:
        return f"❌ Error extracting forecast: {str(e)}"


@mcp.tool()
async def get_weather_page_content_israel() -> str:
    """Extract and clean weather page content for LLM context (RAG).
    
    This tool extracts all readable content from the currently loaded weather page,
    cleans it up (removes noise, extra whitespace), and returns it formatted for
    the LLM to use as context to answer questions about the weather.
    
    Returns the cleaned page content suitable for RAG (Retrieval-Augmented Generation).
    """
    try:
        _, page = await ensure_browser_open()
        
        # Get the full page content with better cleanup
        page_content = await page.evaluate("""
            () => {
                // Remove unwanted elements
                const elementsToRemove = document.querySelectorAll('script, style, noscript, meta, link');
                elementsToRemove.forEach(elem => elem.remove());
                
                // Get text from main content areas
                let text = '';
                
                // Try to get main content
                const mainContent = document.querySelector('main') || 
                                  document.querySelector('[role="main"]') ||
                                  document.querySelector('.container') ||
                                  document.querySelector('.content') ||
                                  document.body;
                
                if (mainContent) {
                    text = mainContent.innerText;
                } else {
                    text = document.body.innerText;
                }
                
                // Clean up excessive whitespace
                text = text.replace(/\\n\\s*\\n\\s*\\n/g, '\\n\\n');  // Multiple newlines
                text = text.replace(/\\s{2,}/g, ' ');  // Multiple spaces
                text = text.trim();
                
                return text;
            }
        """)
        
        if not page_content or page_content.strip() == "":
            return "⚠️ No content found on the weather page. The page may not have loaded correctly."
        
        # Additional cleaning and structuring
        cleaned_content = []
        lines = page_content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely noise
            if len(line) > 3:
                cleaned_content.append(line)
        
        # Join and limit to reasonable size for LLM context
        result_text = '\n'.join(cleaned_content[:100])  # Limit to first 100 lines
        
        # Format nicely for LLM consumption
        formatted_result = f"""
📄 **Israeli Weather Page Content (RAG Context)**
═══════════════════════════════════════════════

{result_text}

═══════════════════════════════════════════════
⚡ This content is extracted from {page.url} and can be used 
as context to answer weather-related questions about Israel.
"""
        
        return formatted_result
        
    except Exception as e:
        return f"❌ Error extracting page content: {str(e)}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
