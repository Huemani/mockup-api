[README.md](https://github.com/user-attachments/files/25081891/README.md)
# Mockup Displacement API

Photoshop-quality displacement mapping for t-shirt mockups using OpenCV.

## Features

- **Bicubic interpolation** - Smooth displacement without jagged edges
- **Automatic displacement map generation** - From any base image
- **Alpha channel support** - Transparent PNGs work perfectly
- **Position, scale, rotation** - Full control over design placement
- **Fast processing** - Typically < 1 second

## API Endpoints

### `POST /generate-mockup`

Generate a mockup with displacement mapping.

**Request Body:**
```json
{
  "baseImageUrl": "https://res.cloudinary.com/.../base.jpg",
  "designImageUrl": "https://res.cloudinary.com/.../design.png",
  "position": {"x": 150, "y": 200},
  "scale": 0.8,
  "rotation": 0,
  "displacementStrength": 15,
  "blendMode": "normal"
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `baseImageUrl` | string | ✅ | - | URL to base t-shirt image |
| `designImageUrl` | string | ✅ | - | URL to design image (PNG recommended) |
| `position.x` | number | ✅ | - | X position on base image |
| `position.y` | number | ✅ | - | Y position on base image |
| `scale` | number | ❌ | 1.0 | Design scale factor |
| `rotation` | number | ❌ | 0 | Rotation in degrees |
| `displacementStrength` | number | ❌ | 15 | Displacement intensity (0-100) |
| `blendMode` | string | ❌ | "normal" | "normal" or "multiply" |

**Response:**
```json
{
  "success": true,
  "mockupUrl": "https://your-api.railway.app/output/abc123.jpg",
  "processingTime": 0.82
}
```

### `GET /output/<filename>`

Retrieve generated mockup images.

### `GET /health`

Health check endpoint.

### `POST /generate-dispmap`

Generate only the displacement map (for debugging).

## Deployment to Railway

### Option 1: GitHub Deploy

1. Push this code to a GitHub repository
2. Go to [railway.app](https://railway.app)
3. Create new project → Deploy from GitHub repo
4. Railway auto-detects Python and deploys

### Option 2: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize and deploy
railway init
railway up
```

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port (Railway sets this) |
| `OUTPUT_DIR` | /tmp/outputs | Directory for generated images |
| `FLASK_DEBUG` | false | Enable debug mode |

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Server starts at `http://localhost:5000`

## Testing

```bash
# Test with curl
curl -X POST http://localhost:5000/generate-mockup \
  -H "Content-Type: application/json" \
  -d '{
    "baseImageUrl": "https://res.cloudinary.com/ducsuev69/image/upload/v1234/base.jpg",
    "designImageUrl": "https://res.cloudinary.com/ducsuev69/image/upload/v1234/design.png",
    "position": {"x": 150, "y": 200},
    "scale": 0.8,
    "displacementStrength": 15
  }'
```

## How It Works

1. **Download images** from provided URLs
2. **Generate displacement map** from base image:
   - Convert to grayscale
   - Apply Gaussian blur (smooth transitions)
   - Enhance contrast with CLAHE
3. **Process design**: scale and rotate
4. **Extract displacement region** matching design position
5. **Apply displacement** using OpenCV `remap()` with bicubic interpolation
6. **Composite** displaced design onto base image
7. **Return** URL to generated mockup

### Why This Works Better Than PixiJS

| PixiJS | OpenCV |
|--------|--------|
| 8-bit pixel shifting | Float32 coordinate mapping |
| WebGL texture limits | Full CPU precision |
| Client-side (slow) | Server-side (fast) |
| Jagged edges | Bicubic interpolation |

## Frontend Integration (Lovable)

```typescript
const API_URL = 'https://your-api.railway.app';

async function generateMockup(
  baseImageUrl: string,
  designImageUrl: string,
  position: { x: number; y: number },
  scale: number,
  displacementStrength: number = 15
) {
  const response = await fetch(`${API_URL}/generate-mockup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      baseImageUrl,
      designImageUrl,
      position,
      scale,
      displacementStrength
    })
  });

  const data = await response.json();

  if (data.success) {
    return data.mockupUrl;
  } else {
    throw new Error(data.error);
  }
}
```

## License

MIT
