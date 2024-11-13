curl 'https://www.petfinder.com/search/?page=1&limit\[\]=100&status=adoptable&token=PTJq6Ve9I8sWeWAxmdhd0UeAdHqYm33Ow0ySMhgXgmA&distance\[\]=100&type\[\]=dogs&include_transportable=true' \
  -H 'accept: application/json' \
  -H 'x-requested-with: XMLHttpRequest' \
| jq ' .result.animals.[].animal | { id: .id, name: .name, breed: .breeds_label, age: .age, sex: .sex, size: .size }'