curl 'https://www.petfinder.com/search/?page=1&limit\[\]=40&status=adoptable&token=WBElNdrUlFLsAMgM61QT5l7YiQLbcDEUv-9vAv6MHi8&location_slug[]=us&type\[\]=dogs&include_transportable=true' \
  -H 'accept: application/json, text/plain, */*' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'priority: u=1, i' \
  -H 'referer: https://www.petfinder.com/search/dogs-for-adoption/us/ny/11238/' \
  -H 'sec-ch-ua: "Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-origin' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36' \
  -H 'x-requested-with: XMLHttpRequest' \
  | jq '.result.animals | map(select(.primary_photo != null)) | length'
