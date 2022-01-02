FROM node:16-alpine

WORKDIR /app
COPY package.json ./
COPY yarn.lock ./

USER node

RUN yarn install

COPY . ./

CMD [ "echo", "hello world" ]