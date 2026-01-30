async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        const data = await response.json();

        document.getElementById("status").innerText =
            `Backend status: ${data.status}`;
    } catch (error) {
        document.getElementById("status").innerText =
            "Backend not reachable";
    }
}
